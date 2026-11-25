#! /usr/bin/env python

import os
os.environ['CRDS_PATH'] = os.path.join(os.environ['HOME'], 'crds_cache')
os.environ['CRDS_SERVER_URL'] = 'https://roman-crds.stsci.edu/'
import json
import numpy as np
import re
import warnings

import astropy.coordinates as ac
import astropy.table as at
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from typing import Optional

import asdf
import crds
import roman_datamodels as rdm
from romancal.associations import asn_from_list
from romancal.pipeline import MosaicPipeline


def update_fits_header_from_meta(key_dict, header, meta):
    '''Update FITS header with values from asdf meta.
    
    Inputs
    ------
    key_dict : dict
        Dictionary mapping asdf meta keywords to FITS header keys.
    header : fits header
        FITS header object to be updated.
    meta : asdf meta
        asdf metadata
    '''
    for asdf_key, fits_key in key_dict.items():
        if asdf_key in meta.keys():
            if fits_key == 'CRDS':
                crds = meta[asdf_key]
                header.set('CRDS_CTX', crds['context'])
                header.set('CRDS_VER', crds['version'])
            elif type(meta[asdf_key]) == Time:
                header.set(fits_key, meta[asdf_key].mjd)
            else:
                header.set(fits_key, meta[asdf_key])
        else:
            header.set(fits_key, '', comment='Not found in asdf file')
    return header

def calc_pix_area(wcs):
    # ripped from stsci.skypac.pamutils._compute_pam_sd
    # https://stsci-skypac.readthedocs.io/en/latest/source/pamutils.html
    shape = wcs.array_shape
    x = np.arange(1, shape[1]+1, dtype='float') - wcs.sip.crpix[0]
    y = np.arange(1, shape[0]+1, dtype='float') - wcs.sip.crpix[1]
    ar = np.arange(wcs.sip.a_order + 1)
    br = np.arange(wcs.sip.b_order + 1)
    ones_a = np.ones(wcs.sip.a_order + 1)
    ones_b = np.ones(wcs.sip.b_order + 1)
    # "coordinate vectors" (e.g., (1, x, x**2, x**3, ...)) used in
    # distortion bilinear forms:
    ax = np.outer(x, ones_a)**ar
    ay = np.outer(y, ones_a)**ar
    bx = np.outer(x, ones_b)**br
    by = np.outer(y, ones_b)**br
    # derivatives of the "coordinate vectors" with regard to x & y:
    adx = np.roll(ax, 1, 1) * ar
    ady = np.roll(ay, 1, 1) * ar
    bdx = np.roll(bx, 1, 1) * br
    bdy = np.roll(by, 1, 1) * br
    # derivatives of the binomial forms:
    A = wcs.sip.a.T
    B = wcs.sip.b.T
    dadx = 1.0 + np.tensordot(ay.T, np.tensordot(A, adx, (1, 1)), (0, 0))
    dady = np.tensordot(ady.T, np.tensordot(A, ax, (1, 1)), (0, 0))
    dbdx = np.tensordot(by.T, np.tensordot(B, bdx, (1, 1)), (0, 0))
    dbdy = 1.0 + np.tensordot(bdy.T, np.tensordot(B, bx, (1, 1)), (0, 0))
    # compute rescaled Jacobian
    jacobian = np.abs(dadx * dbdy - dady * dbdx)
    return jacobian

def asdf_to_fits(im, json_file, multiply_pam=True, multiply_exptime=True, add_dumo=True):
    '''Convert Roman L2 image datamodel asdf format to FITS.
    
    Inputs
    ------
    im : roman datamodel
        Roman L2 image or romanisim output
    json_file : path-like
        Path to JSON file with mapping of metadata to FITS keywords.
        
    Returns
    -------
    hdulist : fits.HDUList
        FITS HDUList in dolphot-ready format (equivalent to romanmask output).
    '''
    # asdf to fits keywords
    with open(json_file, 'r') as f:
        key_map = json.load(f) 
    ny, nx = im.data.shape
    sip_header = im.meta.wcs.to_fits_sip(bounding_box=((-0.5, nx - 0.5), (-0.5, ny - 0.5)))
    data = im.data.copy()
    if add_dumo and hasattr(im, 'dumo'):
        data = data + dumo
    hdu = fits.PrimaryHDU(header=sip_header, data=im.data.copy())
    hdu.header.set('EXTNAME', 'DATA')
    hdu.header.set('BUNIT', 'DN/s', 'Image units')
    # relevant metadata
    for key in key_map.keys():
        if hasattr(im.meta, key):
            hdu.header = update_fits_header_from_meta(key_map[key], hdu.header, getattr(im.meta, key))
    # JANKY
    if hasattr(im.meta, 'ref_file'):
        if 'crds://' in im.meta.ref_file.gain:
            gainfile = os.path.join(os.environ['CRDS_PATH'], 'references/roman/wfi', 
                                    im.meta.ref_file.gain.split('crds://')[-1])
            with rdm.open(gainfile) as gn:
                hdu.header.set('GAIN', np.nanmean(gn.data[4:-4, 4:-4]))
        else:
            hdu.header.set('GAIN', rparam.reference_data['gain'])
        if 'crds://' in im.meta.ref_file.readnoise:
            rnfile = os.path.join(os.environ['CRDS_PATH'], 'references/roman/wfi', 
                                im.meta.ref_file.readnoise.split('crds://')[-1])
            with rdm.open(rnfile) as rn:
                hdu.header.set('RDNOISE', np.nanmean(rn.data[4:-4, 4:-4]))
        else:
            hdu.header.set('RDNOISE', rparam.reference_data['readnoise'])
        if multiply_pam:
            crds_param = {'ROMAN.META.INSTRUMENT.DETECTOR': im.meta.instrument.detector,
                          'ROMAN.META.INSTRUMENT.NAME': im.meta.instrument.name,
                          'ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT' : im.meta.instrument.optical_element,
                          'ROMAN.META.EXPOSURE.TYPE' : im.meta.exposure.type,
                          'ROMAN.META.EXPOSURE.START_TIME': im.meta.exposure.start_time.isot}
            area_ref = None
            try:
                reffiles = crds.getreferences(crds_param, observatory='roman', 
                                              reftypes=['area'], # , 'readnoise', 'gain'
                                              context=im.meta.ref_file.crds.context,
                                              ignore_cache=False, fast=True)
                area_ref = reffiles['area']
            except Exception:
                print('Failed to acquire pixel area reference file.')
            if area_ref is not None:
                pamfile = rdm.open(area_ref)
                pam = pamfile.data
            else:
                pam = calc_pix_area(WCS(hdu.header))
                area_ref = 'calculated from WCS'
            hdu.data *= pam
            hdu.header.set('HISTORY', f'Multiplied by pixel area map ({area_ref.split(os.sep)[-1]})')
    if 'MID_TIME' in hdu.header.keys():
        if hdu.header['MID_TIME'] == '':
            hdu.header.set('MID_TIME', (hdu.header['EXPSTART'] + hdu.header['EXPEND']) / 2.)
        hdu.header.set('MJD-OBS', hdu.header['MID_TIME'])
    hdu.header.set('AIRMASS', 0.0)
    if 'EFFTIME' in hdu.header.keys():
        hdu.header.set('EXPTIME0', hdu.header['EFFTIME'])
        if multiply_exptime:
            hdu.data *= hdu.header['EFFTIME']
            hdu.header.set('BUNIT', 'DN', 'Image units')
            hdu.header.set('HISTORY', f'Multiplied by exposure time (EFFTIME, {hdu.header["EFFTIME"]})')
    if hasattr(im, 'dq'):
        # mask_sat = (im.dq & 2) > 0
        # mask_bad = (im.dq & 1+8+1024) > 0
        unmasked = hdu.data[(im.dq == 0) & np.isfinite(hdu.data)]
        bad_val = min(np.abs(unmasked.min()) * -1.1, -100.)
        sat_val = max(unmasked.max() * 1.1, 99999999999.)
        # hdu.data[mask_bad] = bad_val
        # hdu.data[mask_sat] = sat_val
        hdu.header.set('BADPIX', bad_val)
        hdu.header.set('SATURATE', sat_val)
    if ('PHOTMJSR' in hdu.header.keys()) and ('PIXAREA' in hdu.header.keys()):
        cps_to_mjy = hdu.header['PHOTMJSR'] * hdu.header['PIXAREA'] * 1e6
        hdu.header.set('DOL_C2JY', -2.5 * np.log10(cps_to_mjy))
    else:
        hdu.header.set('DOL_C2JY', 0)
    hdu.header.set('DOL_ROMN', 0)
    hdulist = fits.HDUList([hdu])
    return hdulist

def make_l3(l2_list, product_name, asn_kwargs=dict(target='none'),
            configure_log=False, on_disk=True, save_results=True,
            flux_args=dict(),
            skymatch_args=dict(lower=-100, upper=1000, skymethod='global+match'),
            outlier_args=dict(),
            resample_args=dict(), 
            source_catalog_args=dict(skip=True)):
    asn = asn_from_list.asn_from_list([(im, 'science') for im in l2_list],
                                      product_name=product_name, 
                                      with_exptype=True, **asn_kwargs)
    result = MosaicPipeline.call(asn, 
                                 configure_log=configure_log, 
                                 on_disk=on_disk, 
                                 save_results=save_results,
                                 steps={'flux': flux_args,
                                        'skymatch': skymatch_args,
                                        'outlier_detection': outlier_args,
                                        'resample': resample_args,
                                        'source_catalog' : source_catalog_args})
    return result
