#! /usr/bin/env python

import os
os.environ['CRDS_PATH'] = os.path.join(os.environ['HOME'], 'crds_cache')
os.environ['CRDS_SERVER_URL'] = 'https://roman-crds.stsci.edu/'
import copy
import importlib
import json
import shutil
import numpy as np
import pandas as pd
import re
import s3fs
import time
import warnings

import astropy.coordinates as ac
import astropy.table as at
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy import units as u
from dataclasses import dataclass
from typing import Optional
from mhealpy import HealpixMap
from pathlib import Path

import asdf
import crds
import galsim
import hpgeom
import pysiaf
import roman_datamodels as rdm
import romanisim
if romanisim.__version__ >= '0.14.0':
    import romanisim.models.parameters as rparam
    import romanisim.models.wcs as rwcs
else:
    import romanisim.parameters as rparam
    import romanisim.wcs as rwcs
import romanisim.persistence as rpersist
from romanisim import ris_make_utils
from romanisim.util import random_points_in_cap
from romancal.associations import asn_from_list
from romancal.pipeline import MosaicPipeline

import vaex
try:
    import galaxia_ananke as galaxia
except ImportError:
    print('Unable to import galaxia_ananke. Not compatible with new pyananke h5 format.')


def read_isim_input_catalogs(hplist, catalog_dir, name_template, catalog_type='isim', 
                             ab_vega_path='input_data/aux/abvega_offset_0002_rmap.csv',
                             **kwargs):
    '''Read HEALPix input catalogs from directory.
    
    Inputs
    ------
    hplist : array-like of int
        List of HEALPix indices
            SOC sims: nest512, Galactic lonlat
            lsstsim: ring256, ICRS lonlat
    catalog_dir : str, path-like
        Directory or S3 path with catalog files.
    name_template : str
        String for catalog name, to be formatted with healpix index.
        'cat-{}.fits' for SOC input catalogs
        'lsstsim_{}_isim.ecsv' for LSST sims
    catalog_type : str, one of 'isim', 'pyananke'
        Is the input catalog in isim-ready format or does additional 
        preprocessing need to be done? (Additional preprocessing currently 
        implemented only for pyananke files.)
    kwargs : dict
        Additional keyword arguments to be passed to table reader function. 
        Recommended:
        `format='ecsv', engine='pyarrow'` for ecsv;
        `format='parquet'` for parquet;
        `format='fits'` for SOC inputs
        
    Returns
    -------
    t : astropy.table.Table
        Stack of all input catalogs found corresponding to HEALPix indices
        in hplist, assumed to be in romanisim-ready format.
    '''
    on_s3 = catalog_dir.startswith('s3://')
    if on_s3:
        fs = s3fs.S3FileSystem(anon=True)
    tables = []
    for hpix in hplist:
        table_path = os.path.join(catalog_dir, name_template.format(hpix))
        table_exists = fs.isfile(table_path) if on_s3 else os.path.isfile(table_path)
        if not table_exists:
            print(f'No file found at {table_path}')
            continue
        if on_s3:
            with fs.open(table_path, 'rb') as f:
                if catalog_type == 'isim':
                    tables.append(at.Table.read(f, **kwargs))
                elif catalog_type == 'pyananke':
                    ds = vaex.open(f, group=galaxia.STARCATALOG_GROUP)
                    tables.append(pyananke_to_isim(ds, ab_vega_path=ab_vega_path))
        else:
            if catalog_type == 'isim':
                tables.append(at.Table.read(table_path, **kwargs))
            elif catalog_type == 'pyananke':
                ds = vaex.open(table_path, group=galaxia.STARCATALOG_GROUP)
                tables.append(pyananke_to_isim(ds, ab_vega_path=ab_vega_path))
    if len(tables) == 0:
        print(f'No catalogs found at {catalog_dir} for HEALPix {hplist}!')
        return at.Table()
    elif len(tables) == 1:
        t = tables[0]
    else:
        t = at.vstack(tables)
    return t

def pyananke_to_isim(ds, ab_vega_path='input_data/aux/abvega_offset_0002_rmap.csv'):
    '''Convert vaex dataframe of pyananke-simulated stars to romanisim input.
    
    Inputs
    ------
    ds : vaex.dataframe.DataFrame
        Must have columns ra, dec, roman_<filter>
    ab_vega : dict or pd.Series of floats
        Detector-averaged AB-Vega offsets per WFI filter
        
    Returns
    -------
    t : astropy.table.Table
        at.Table of coordinates and fluxes in romanisim-ready format.
    '''
    ab_vega = pd.read_csv(ab_vega_path, index_col=0).mean()
    out_cols = []
    for filt, offset in ab_vega.items():
        ananke_col = f'roman_{filt.lower()}'
        ab_col = f'{filt.upper()}_AB'
        mgy_col = filt.upper()
        if ananke_col in ds.get_column_names(regex=f'^{ananke_col}$'):
            ds[ab_col] = ds[f'{ananke_col} + {offset:.8f}']
            ds[mgy_col] = ds[f'10**({ab_col} / -2.5)']
            out_cols.append(mgy_col)
        else:
            print(f'Filter {filt} not found in pyananke catalog!')
    t = ds[['ra', 'dec'] + out_cols].to_astropy_table()
    t['type'] = 'PSF'
    return t

def egg_to_romanisim(input_table, ra, dec, radius=np.pi**-0.5, use_bulge=True):
    # If input is a string (a file), read it in as an astropy table
    if type(input_table) == str:
        df = pd.read_csv(input_table, sep=r'\s+').sample(frac=0.281**-1, replace=True)
        input_table = at.Table.from_pandas(df) #read(input_table, format='pandas.csv', sep=' ')
        
    coord = ac.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    N_gal = len(input_table)
    names = ('ra', 'dec', 'type', 'n', 'half_light_radius', 'pa', 'ba')
    mag_names = ('F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F146', 'F213')
    
    # Shared properties for bulge and disk
    sampled_coords = random_points_in_cap(coord, radius, N_gal)
    ra = sampled_coords.ra.degree
    dec = sampled_coords.dec.degree
    galtype = np.repeat('SER', N_gal)
    PA = np.random.uniform(-180, 180, N_gal)

    # Make table of disk fluxes in maggies
    disk_mags = at.Table([np.power(10, input_table[f'mag_obs_disk_{i}_AB'].data / -2.5) for i in mag_names],
                           names=mag_names)
    r50_disk = input_table['R50_disk_arcsec']
    ba_disk = input_table['disk_ratio']
    n_disk = np.repeat(1.0, N_gal) # Sersic index of 1 for disk
    disk_table = at.Table([ra, dec, galtype, n_disk, r50_disk, PA, ba_disk],
                       names=names)
    # Concatenate disk properties and magnitudes into one table
    disk_table = at.hstack([disk_table, disk_mags])
    if use_bulge: # inject bulge and disk separately, copy  
        bulge_mags = at.Table([np.power(10, input_table[f'mag_obs_bulge_{i}_AB'].data / -2.5) for i in mag_names],
                           names=mag_names)
        r50_bulge = input_table['R50_bulge_arcsec']
        ba_bulge = input_table['bulge_ratio']
        n_bulge = np.repeat(4.0, N_gal) # Sersic index of 4 for bulge
        bulge_table = at.Table([ra, dec, galtype, n_bulge, r50_bulge, PA, ba_bulge],
                           names=names)
        # Concatenate bulge properties and magnitudes into one table
        bulge_table = at.hstack([bulge_table, bulge_mags])
        # Remove bulges fainter than faintest disk
        bulge_table = bulge_table[bulge_table['F087'] > np.min(disk_table['F087'])]
        # Combine bulge and disk tables
        combined_table = at.vstack([disk_table, bulge_table])
        return combined_table
    else:
        return disk_table

@dataclass(init=True, repr=True)
class PointWFI:
    """Adapted from Romanisim demo notebook:
    github.com/spacetelescope/roman_notebooks/blob/main/notebooks/romanisim/romanisim.ipynb
    
    Inputs
    ------
    ra : float, default 0.0
        Right ascension of the WFI focal plane array center (SIAF aperture 
        name WFI_CEN).
    dec : float, default 0.0
        Declination of the WFI focal plane array center.
    pa_idl : float, default 0.0
        Position angle of the WFI ideal axis in deg E of N. A value of 0.0d 
        would place the WFI in the "smiley face" orientation (U-shaped) on the 
        celestial sphere. To place WFI such that the position angle of the V3 
        axis is pointing north, use a position angle of -60d.
    apername : str, default 'WFI_CEN'
        Name of SIAF aperture coordinates are specified for; should be WFI_CEN 
        for full-frame observations, otherwise the name of the SCA to be 
        simulated ("WFI<NN>_FULL").

    Description
    -----------
    To use this class, instantiate it with your initial pointing like so:
        >>> point = PointWFI(ra=30, dec=-45, position_angle=10)
    and then dither using the dither method:
        >>> point.dither(x_offset=10, y_offset=140)
    Dither shifts are in are in the ideal coordinate system of the WFI, i.e, 
    with the WFI oriented in a U-shape with +x to the right and +y up.
    """
    # Set default pointing parameters
    ra: float = 0.0
    dec: float = 0.0
    pa_idl: float = 0.0
    ref_apername : str = 'WFI_CEN'
    # Post init method sets some other defaults and initializes
    # the attitude matrix using PySIAF.
    def __post_init__(self) -> None:
        self.siaf = pysiaf.Siaf('Roman')
        self.ref_aper = self.siaf[self.ref_apername]
        self.v2_ref = self.ref_aper.V2Ref
        self.v3_ref = self.ref_aper.V3Ref
        self.pa_v3 = self.pa_idl - self.ref_aper.V3IdlYAngle
        self.att = pysiaf.utils.rotations.attitude(self.v2_ref, self.v3_ref,
                                                   self.ra, self.dec, self.pa_v3)
        for apername in self.siaf.apernames:
            self.siaf[apername].set_attitude_matrix(self.att)
        # Compute the V3 position angle
        self.tel_roll = pysiaf.utils.rotations.posangle(self.att, 0, 0)

    def dither(self, x_offset: int | float | np.typing.ArrayLike, 
               y_offset: int | float | np.typing.ArrayLike):
        """Shift the telescope pointing by (x_offset, y_offset) arcsec (ideal).

        Inputs
        ------
        x_offset (float): The offset in arcseconds in the ideal X direction.
        y_offset (float): The offset in arcseconds in the ideal Y direction.
        """
        return self.ref_aper.idl_to_sky(x_offset, y_offset)
    
    def siaf_to_healpix(self, apername : str, h5_dir : Optional[str] = None, 
                        nside : int = 256,  inclusive : bool = True, 
                        fact : int = 32, nest : bool = False, 
                        galactic : bool = False, radius : float = 0.1,
                        ) -> np.typing.NDArray:
        '''Get HEALpix indices overlapping with specified SIAF aperture.
        
        Inputs
        ------
        apername : str
            SIAF aperture name ('WFI_CEN' or 'WFI<NN>_FULL')
        nside : int, default 256 
        Number of HEALPix polygon sides (power of 2).
        inclusive : bool, default True
            Find all overlapping HEALPix or just those that directly intersect 
            the polygon edges?
        fact : int, default 32
            Factor to increase resolution by when evaluating HEALPix overlaps.
        nest : bool, default False
            Use nested HEALPix schema?
        galactic : bool, default False
            Whether SIAF sky coordinates need to be converted to Galactic.
        radius : float, default 0.1
            Search radius in deg if specified aperture does not have defined 
            corners, in which case the sky reference point is used.
        '''
        aper = self.siaf[apername]
        try:
            lon, lat = aper.corners('sky')
        except TypeError:
            lon, lat = aper.reference_point('sky')
            lon, lat = np.array([lon]), np.array([lat])
        if galactic:
            coord = ac.SkyCoord(lon, lat, unit='deg', frame='icrs').transform_to('galactic')
            lon, lat = coord.l.deg, coord.b.deg
        if h5_dir:
            cwd = Path(h5_dir)
            state_file = list(cwd.glob('*.h5'))[0].with_suffix('').with_suffix('.state')
            all_healpix = np.sort([
                int(foo[0])
                for f in cwd.glob(state_file.with_suffix('.*.h5').name)
                if (foo:=re.findall(r".*\.(\d+)\.h5", f.name))
                ])
            hp_map = HealpixMap(None, all_healpix, density=True)
            vertices = ac.SkyCoord(lon, lat, unit='deg', frame=['icrs', 'galactic'][int(galactic)])
            hplist = hp_map.uniq[hp_map.query_polygon(vertices.cartesian.xyz.T.value, inclusive=True)]
        else:
            if len(lon) == 1:
                hplist = hpgeom.query_circle(nside, lon, lat, radius, inclusive=inclusive,
                                            fact=fact, nest=nest, lonlat=True, degrees=True)
            else:
                hplist = hpgeom.query_polygon(nside, lon, lat, inclusive=inclusive,
                                            fact=fact, nest=nest, lonlat=True, degrees=True)
        return hplist

def expand_dithers(row, auxfiles_dir='/home/mdurbin/pipelines/input_data/aux/',
                   gap_dither_column='DITHER_GAP', sub_dither_colname='DITHER_SUB'):
    # ref_apername = 'WFI_CEN' if (row.SCA < 0) else f'WFI{row.SCA:02d}_FULL'
    initial_pointing = PointWFI(row.RA, row.DEC, pa_idl=row.PA, ref_apername='WFI_CEN')
    dither_gap, dither_sub = None, None
    if row.DITHER_GAP.upper() != "NONE":
        dither_gap = read_dithers(os.path.join(auxfiles_dir, 'WfiImagingGap.txt'))[row.DITHER_GAP.upper()]
    if row.DITHER_SUB.upper() != "NONE":
        dither_sub = read_dithers(os.path.join(auxfiles_dir, 'WfiImagingSubpixel.txt'))[row.DITHER_SUB.upper()]
    if (dither_gap is None) and (dither_sub is None):
        dither_out = pd.DataFrame(data=[[1, 0.0, 0.0]], columns=['EXPOSURE', 'xshift', 'yshift'])
    elif (dither_gap is not None) and (dither_sub is None):
        dither_out = dither_gap.rename(columns={'dither_number':'EXPOSURE'})
    elif (dither_gap is None) and (dither_sub is not None):
        dither_out = dither_sub.rename(columns={'dither_number':'EXPOSURE'})
    elif (dither_gap is not None) and (dither_sub is not None):
        dither_gap = dither_gap.set_index('dither_number').rename_axis('gap')
        dither_sub = dither_sub.set_index('dither_number').rename_axis('sub')
        dither_all = pd.DataFrame(index=pd.MultiIndex.from_product([dither_gap.index, dither_sub.index]),
                                  columns=['EXPOSURE'], dtype='int').join(dither_gap, on='gap').add(dither_sub, level='sub')
        dither_all['EXPOSURE'] = dithers_all['EXPOSURE'].fillna(1).astype('int').cumsum()
        dither_out = dither_all.reset_index(drop=True)
    new_ra, new_dec = initial_pointing.dither(dither_out['xshift'], dither_out['yshift'])
    dithered_plan = pd.concat([row.drop(['DITHER_GAP', 'DITHER_SUB'])] * len(dither_out), 
                              axis=1, ignore_index=True).T.join(dither_out[['EXPOSURE']])
    dithered_plan['RA'] = new_ra
    dithered_plan['DEC'] = new_dec
    return dithered_plan

def read_obs_plan(ecsvfile, auxfiles_dir='/home/mdurbin/pipelines/input_data/aux/'):
    obs_plan_input = at.Table.read(ecsvfile, format='ecsv', engine='pyarrow').to_pandas()
    if 'SCA' not in obs_plan_input.columns:
        obs_plan_input['SCA'] = -1
    has_exposure = 'EXPOSURE' in obs_plan_input.columns
    has_gap_dither = 'DITHER_GAP' in obs_plan_input.columns
    has_sub_dither = 'DITHER_SUB' in obs_plan_input.columns
    if not has_exposure:
        if not has_gap_dither:
            obs_plan_input['DITHER_GAP'] = 'NONE'
        if not has_sub_dither:
            obs_plan_input['DITHER_SUB'] = 'NONE'
        obs_plan = pd.concat([expand_dithers(row, auxfiles_dir) for _, row 
                              in obs_plan_input.iterrows()], ignore_index=True)
    else:
        obs_plan = obs_plan_input
    expand_bandpass = obs_plan['BANDPASS'].eq('ALL').any()
    expand_sca = obs_plan['SCA'].lt(0).any()
    if expand_bandpass:
        obs_plan['BANDPASS'] = obs_plan['BANDPASS'].replace('ALL', 'F062,F087,F106,F129,F146,F158,F184,F213').\
            str.split(',', expand=False)
        obs_plan = obs_plan.explode('BANDPASS', ignore_index=True)
    if expand_sca:
        obs_plan['SCA'] = obs_plan['SCA'].replace(-1, ','.join(map(str, range(1, 19)))).\
            str.split(',', expand=False)
        obs_plan = obs_plan.explode('SCA', ignore_index=True)
        obs_plan['SCA'] = pd.to_numeric(obs_plan['SCA'])
    return obs_plan

def read_dithers(dither_file : str | os.PathLike):
    # read in subpixel dither pattern spec file into dict of dataframes
    # https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-dithering
    dithers = pd.read_csv(dither_file, header=None, sep=r'\s+', 
                          names=['dither_number', 'xshift', 'yshift'])
    groups = dithers['dither_number'].str.contains('^[A-Z]').cumsum()
    tables = {g.iloc[0,0]: g.iloc[1:].transform(pd.to_numeric).reset_index(drop=True) 
              for k, g in dithers.groupby(groups)}
    return tables

def set_obs_metadata(meta, program : int, plan : int, passnum : int, 
                     seg : int, obs : int, visit : int, exposure : int):
    obs_meta = {'program'        : program,
                'execution_plan' : plan,
                'pass'           : passnum,
                'segment'        : seg,
                'observation'    : obs,
                'visit'          : visit,
                'exposure'       : exposure,
                }
    visit_id = f'{program:05d}{plan:02d}{passnum:03d}{seg:03d}{obs:03d}{visit:03d}'
    observation_id = f'{visit_id}{exposure:04d}'
    obs_meta['visit_id'] = visit_id
    obs_meta['observation_id'] = observation_id
    meta.update(obs_meta)
    return meta

def make_l2_filename(meta, visit_id=None, exposure=None, detector=None, opt_elem=None):
    # https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products
    visit_id = meta.observation.visit_id if visit_id is None else visit_id
    exposure = meta.observation.exposure if exposure is None else exposure
    detector = meta.instrument.detector if detector is None else detector
    opt_elem = meta.instrument.optical_element if opt_elem is None else opt_elem
    filename = f'r{visit_id}_{exposure:04d}_{detector.lower()}_{opt_elem.lower()}_cal.asdf'
    return filename

def make_l2(t : at.Table, ra_cen : float, dec_cen : float, 
            bandpass : str, ma_table_number : int, sca : int, 
            pa_cen : float = 0.0, obs_date : str = '2027-06-01T00:00:00', 
            psftype : str = 'epsf', seed : int = 7, 
            usecrds : bool = True, persist : Optional[rpersist.Persistence] = None, 
            chromatic : bool = False, variable_psf : bool = True,
            ):
    '''Simulate single-SCA L2 frame with romanisim.
    
    Inputs
    ------
    t : at.Table
        Astropy table of source information.
    ra_cen : float
        RA of WFI_CEN aperture reference point.
    dec_cen : float
        Dec of WFI_CEN aperture reference point.
    bandpass : str
        Name of imaging filter to be simulated.
    ma_table_number : int
        Multiaccum table number that specifies exposure readout pattern.
        https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-multiaccum-ma-tables
    sca : int
        WFI SCA (sensor chip assembly) number, 1-18.
    pa_cen : float, default 0.0
        Position angle of WFI_CEN ideal axis, in deg E of N.
    obs_date : str, default '2027-06-01T00:00:00'
        ISOT-format date of exposure start.
    psftype : str, default 'epsf'
        Type of PSF; one of 'epsf', 'stpsf', 'galsim'.
    seed : int, default 7
        Random number generator seed.
    usecrds : bool, default True
        Whether to use CRDS reference files. If true, requires CRDS_PATH and 
        CRDS_SERVER_URL environment variables to be set.
    persist : (optional) rpersist.Persistence, default None
        romanisim Persistence object or None.
    chromatic : bool, default False
        Whether to use a chromatic PSF. Not fully implemented yet.
    variable_psf : bool, default True
        Allows for fast point source placement via PSF interpolation.
        Incompatible with chromatic = True.
    '''
    # turn off asdf version warnings
    warnings.filterwarnings('ignore', category=asdf.exceptions.AsdfPackageVersionWarning)
    if usecrds:
        for k in rparam.reference_data.keys():
            rparam.reference_data[k] = None
    # initialize isim image with no sources
    metadata = ris_make_utils.set_metadata(date=obs_date, bandpass=bandpass, sca=sca, 
                                           ma_table_number=ma_table_number, usecrds=usecrds)
    rwcs.fill_in_parameters(metadata, ac.SkyCoord(ra_cen, dec_cen, unit='deg', frame='icrs'), 
                            boresight=False, pa_aper=pa_cen)
    rng = galsim.UniformDeviate(seed)
    im, extras = romanisim.image.simulate(metadata, at.Table(), usecrds=usecrds, psftype=psftype, 
                                          level=2, persistence=persist, rng=rng)
    if usecrds:
        importlib.reload(rparam)
    if len(t) == 0:
        print('Zero-length input table; skipping source injection step.')
        return im
    # inject sources that are within the image footprint
    x, y = im.meta.wcs.invert(t['ra'], t['dec'], with_bounding_box=True)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() == 0:
        print(f'No input sources overlap with SCA {sca:02d}; skipping source injection.')
        return im
    print(f'{keep.sum()} sources out of {len(keep)} in WFI{sca:02d} footprint')
    psf = romanisim.psf.make_psf(sca, bandpass, wcs=rwcs.GWCS(im.meta.wcs), variable=variable_psf,
                                 chromatic=chromatic, psftype=psftype, date=obs_date)
    iminj = romanisim.image.inject_sources_into_l2(im, t[keep], x=x[keep], y=y[keep], psf=psf, 
                                                   psftype=psftype, seed=seed, rng=rng)
    return iminj

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
            header.set(fits_key, None, comment='Not found in asdf file')
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

def l2_asdf_to_fits(im, json_file):
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
    img_hdu = fits.PrimaryHDU(header=sip_header, data=im.data)
    img_hdu.header.set('EXTNAME', 'DATA')
    img_hdu.header.set('BUNIT', 'DN', 'Image units')
    pri_hdu = img_hdu
    # pri_hdu = fits.PrimaryHDU()
    # relevant metadata
    for key in key_map.keys():
        if hasattr(im.meta, key):
            pri_hdu.header = update_fits_header_from_meta(key_map[key], pri_hdu.header, getattr(im.meta, key))
    # JANKY
    if 'crds://' in im.meta.ref_file.gain:
        gainfile = os.path.join(os.environ['CRDS_PATH'], 'references/roman/wfi', 
                                im.meta.ref_file.gain.split('crds://')[-1])
        with rdm.open(gainfile) as gn:
            # gn_mean = np.nanmean(gn.data[4:-4, 4:-4])
            # img_hdu.data *= gn.data[4:-4, 4:-4] / gn_mean
            img_hdu.header.set('GAIN', np.nanmean(gn.data[4:-4, 4:-4]))
    else:
        img_hdu.header.set('GAIN', rparam.reference_data['gain'])
    # img_hdu.header.set('GAIN', 1.0)
    if 'crds://' in im.meta.ref_file.readnoise:
        rnfile = os.path.join(os.environ['CRDS_PATH'], 'references/roman/wfi', 
                              im.meta.ref_file.readnoise.split('crds://')[-1])
        with rdm.open(rnfile) as rn:
            img_hdu.header.set('RDNOISE', np.nanmean(rn.data[4:-4, 4:-4]))
    else:
        img_hdu.header.set('RDNOISE', rparam.reference_data['readnoise'])
    
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
        print('Failed to acquire reference file(s).')
    if area_ref is not None:
        pamfile = rdm.open(area_ref)
        pam = pamfile.data
    else:
        pam = calc_pix_area(WCS(img_hdu.header))
    img_hdu.data *= pam * img_hdu.header['EFFTIME']
    mask_sat = (im.dq & 2) > 0
    mask_bad = (im.dq & 1+8+1024) > 0
    bad_val = min(img_hdu.data[~(mask_bad | mask_sat)].min() * 1.1, -100.)
    sat_val = max(img_hdu.data[~(mask_sat | mask_bad)].max() * 1.1, 65536.)
    img_hdu.data[mask_bad] = bad_val
    img_hdu.data[mask_sat] = sat_val
    pri_hdu.header.set('BADPIX', bad_val)
    pri_hdu.header.set('SATURATE', sat_val)
    pri_hdu.header.set('MJD-OBS', pri_hdu.header['MID_TIME'])
    pri_hdu.header.set('AIRMASS', 0.0)
    pri_hdu.header.set('EXPTIME0', pri_hdu.header['EFFTIME'])
    cps_to_mjy = pri_hdu.header['PHOTMJSR'] * pri_hdu.header['PIXAREA'] * 1e6
    pri_hdu.header.set('DOL_C2JY', -2.5 * np.log10(cps_to_mjy))
    pri_hdu.header.set('DOL_ROMN', 0)
    # dq_hdu = fits.ImageHDU(data=im.dq, name='DQ')
    # dq_hdu.header.set('EXTNAME', 'DQ')
    # err_hdu = fits.ImageHDU(data=im.err.astype('float32'), name='ERR')
    # err_hdu.header.set('EXTNAME', 'ERR')
    # var_poisson_hdu = fits.ImageHDU(data=im.var_poisson.astype('float32'), name='VAR_POISSON')
    # var_poisson_hdu.header.set('EXTNAME', 'VAR_POISSON')
    # hdulist = fits.HDUList([pri_hdu, img_hdu, dq_hdu, err_hdu, var_poisson_hdu])
    hdulist = fits.HDUList([pri_hdu])
    return hdulist

def make_l3(product_name, l2_list):
    asn = asn_from_list.asn_from_list([(im, 'science') for im in l2_list],
                                      product_name=product_name, with_exptype=True, 
                                      target='none')
    fname, txt = asn.dump()
    with open(fname, 'w') as f:
        f.write(txt)
    result = MosaicPipeline.call(fname, configure_log=False, on_disk=True, save_results=True,
                                 steps={'skymatch':{'skip': True}, 
                                        'outlier_detection':{'skip':True}, 
                                        'source_catalog':{'skip':True}})
    return result


