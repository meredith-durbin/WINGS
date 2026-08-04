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
import vaex
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

# import asdf
# import crds
# import galsim
# import hpgeom
# import pysiaf
# import roman_datamodels as rdm
# import romanisim
# if romanisim.__version__ >= '0.14.0':
#     import romanisim.models.parameters as rparam
#     import romanisim.models.wcs as rwcs
# else:
#     import romanisim.parameters as rparam
#     import romanisim.wcs as rwcs
# import romanisim.persistence as rpersist
# from romanisim import ris_make_utils
# from romanisim.util import random_points_in_cap
# from romancal.associations import asn_from_list
# from romancal.pipeline import MosaicPipeline

if __name__ == '__main__':
    from romanisim_util import (PointWFI, read_isim_input_catalogs, make_l2, 
                                set_obs_metadata, make_l2_filename, l2_asdf_to_fits)
else:
    from wpipe.romanisim_util import (PointWFI, read_isim_input_catalogs, make_l2,  
                                      set_obs_metadata, make_l2_filename, l2_asdf_to_fits)

def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='new_isim_target', value='*')

def run_isim_single(event_id, my_config):
    my_event = wp.Event(event_id)
    row = my_event.options
    my_params = my_config.parameters
    input_dir = my_params['input_dir']
    catalog_type = my_params['catalog_type']
    aux_dir = my_config['aux_dir']
    point = PointWFI(ra=row['RA'], dec=row['DEC'], pa_idl=row['PA'], ref_apername='WFI_CEN')
    hp_kwargs = dict()
    if catalog_type == 'pyananke':
        h5file = list(Path(input_dir).glob('*.h5'))[0]
        h5root = h5file.with_suffix('').with_suffix('').name
        name_template = h5root + '.{:0' + str(len(h5file.suffixes[-2]) - 1) + 'd}.h5'
        hp_kwargs.update(dict(h5_dir=input_dir))
    else:
        if 'lsstsim' in input_dir:
            name_template = 'lsstsim_{:d}_isim.parquet'
            hp_kwargs.update(dict(nside=256, nest=False, galactic=False))
        elif 'soc_simulations/input_catalogs' in input_dir:
            name_template = 'cat-{:d}.fits'
            hp_kwargs.update(dict(nside=512, nest=True, galactic=True))
    hplist_input = point.siaf_to_healpix(f'WFI{row.SCA:02d}_FULL', **hp_kwargs)
    read_kwargs = dict()
    if '.ecsv' in name_template:
        read_kwargs.update(dict(format='ecsv', engine='pyarrow'))
    elif '.fit' in name_template:
        read_kwargs.update(dict(format='fits'))
    t_input = read_isim_input_catalogs(hplist_input, catalog_dir, name_template, catalog_type=catalog_type, 
                                       ab_vega_path=os.path.join(aux_dir, 'abvega_offset_0002_rmap.csv'),
                                       **read_kwargs)
    hplist_bg = point.siaf_to_healpix(f'WFI{row.SCA:02d}_FULL', nside=256, nest=False, galactic=False)
    if my_params['add_background']:
        t_bg = read_isim_input_catalogs(hplist_bg, my_config['background_dir'], '{:d}.ecsv',
                                        catalog_type='isim', format='ecsv', engine='pyarrow')
        t = at.vstack([t_input, t_bg])
    else:
        t = t_input
    im = make_l2(t, row.RA, row.DEC, row.BANDPASS, row.MA_TABLE_NUMBER, row.SCA, pa_cen=row.PA)
    obs_meta = set_obs_metadata(im.meta.observation, my_config['program_id'], row.PLAN, row.PASS, 
                                row.SEGMENT, row.OBSERVATION, row.VISIT, row.EXPOSURE)
    im.meta.observation.update(obs_meta)
    asdf_filename = make_l2_filename(im.meta)
    im.meta['filename'] = asdf_filename
    fitsfile = l2_asdf_to_fits(im, os.path.join(aux_dir, 'rdm_to_fits_keywords.json'))
    fits_filename = asdf_filename.replace('.asdf', '.fits')
    asdf_path = os.path.join(my_config.procpath, asdf_filename)
    fits_path = os.path.join(my_config.procpath, fits_filename)
    af = asdf.AsdfFile()
    af.tree = {'roman': im}
    af.write_to(asdf_path)
    #im.writeto(asdf_filename, overwrite=True)
    fitsfile.writeto(fits_path, overwrite=True)
    asdf_dp = wp.DataProduct(my_config, filename=asdf_filename, relativepath=my_config.procpath, 
                             group="proc", subtype="isim_asdf_image")
    fits_dp = wp.DataProduct(my_config, filename=fits_filename, relativepath=my_config.procpath, 
                             group="proc", subtype="isim_fits_image")
    # my_job.child_event()
    return

def parse_all():
    parser = wp.PARSER
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_all()
    this_job_id = args.job_id
    this_job = wp.Job(this_job_id)
    this_event = this_job.firing_event
    this_event_id = this_event.event_id
    this_dp_id = this_event.options['dp_id']
    parent_job_id = this_event.parent_job_id
    parent_job = this_event.parent_job
    compname = this_event.options['name']
    # ra_dither = this_event.options['ra_dither']
    # dec_dither = this_event.options['dec_dither']
    print('event', this_event_id, 'dp', this_dp_id)
    # detname = this_event.options['detname']
    catalogID = this_event.options['dp_id']
    catalogDP = wp.DataProduct(catalogID)
    this_conf = catalogDP.config
    # print('DETNAME', detname)
    
    obs_plan = read_obs_plan(target_dp.relativepath, auxfiles_dir=my_config.parameters['aux_dir'])
    for i, row in obs_plan.T.to_dict().items():
        my_event = my_job.child_event('new_isim_run', tag=i,
                                      options={'dp_id': dpid, 'to_run': total, 'name': comp_name,
                                               'submission_type':'scheduler', 'partition': partition,
                                               **row})
        #Should there be a detname key here (line above)?
        my_job.logprint(''.join(["Firing event ", str(my_event.event_id), "  new_isim_run"]))
        my_event.fire()

    checkname = run_stips(this_event_id, this_dp_id, float(ra_dither), float(dec_dither), detname, this_job)
    to_run = this_event.options['to_run']
    this_target = this_conf.target
    #try:
    #    ndetect = my_params['ndetect']
    #except:
    #    this_job.logprint("Couldn't find ndetect parameter, setting to 1")
    #    ndetect = 1
    #if ndetect == 1:
    #    this_job.logprint("ndetect is 1, so setting the detname to the targname")
    #    targname = this_target.name
    #    detname = '.'.join(targname.split('.')[:-1])
    this_job.logprint(''.join(["Grabbing DPS with DETNAME and conf ids of", detname, str(this_conf.config_id), "\n"]))
    print("detname and checkname are ", detname, " and ", checkname)
    #if detname == checkname:
    #    print("SAME")
    #else:
    #    print("FAIL, setting detname to checkname")
    #    detname = checkname
    image_dps = wp.DataProduct.select(config_id=str(this_conf.config_id), data_type="isim_image", subtype=detname)
    #image_dps = wp.DataProduct.select(config_id=str(this_conf.config_id), data_type="isim_image")
    update_option = parent_job.options[compname]
    update_option += 1
    this_job.logprint(''.join(["Got ", str(len(image_dps)), " images \n"]))
    this_job.logprint(''.join(["Completed ", str(update_option), " of ", str(to_run), "\n"]))
    if update_option == to_run:
        if len(image_dps) < to_run:
            raise Exception(f"Lost an image as counts should be {to_run} but is {len(image_dps)}")
        this_job.logprint(''.join(["Completed ", str(update_option), " and to run is ", str(to_run), " firing event\n"]))
        DP = wp.DataProduct(this_dp_id)
        tid = DP.target_id
        path = this_conf.procpath
        #comp_name = 'completed' + this_target.name
        comp_name = 'completed' + detname
        options = {comp_name: 0}
        this_job.options = options
        total = to_run
        #total = len(image_dps)
        # print(image_dps(0))
        for dps in image_dps:
            #print(dps)
            dpid = dps.dp_id
            st = dps.subtype 
            this_job.logprint(''.join(["ID and subtype ", str(dpid), " and ", str(st), "\n"]))
            new_event = this_job.child_event('isim_done', tag=dpid, 
                                             options={'target_id': tid, 'dp_id': dpid, 'submission_type': 'scheduler', 
                                                      'name': comp_name, 'to_run': total, 'detname': detname, 'walltime': '2:00:00'})
            this_job.logprint(''.join(["event detname is ", str(detname)]))
            new_event.fire()
            time.sleep(2)
            #this_job.logprint('isim_done but not firing any events for now\n')
            this_job.logprint(''.join(["Event= ", str(this_event.event_id)]))
        time.sleep(300)
