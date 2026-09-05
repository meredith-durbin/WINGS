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

if __name__ == '__main__':
    from romanisim_util import read_obs_plan, egg_to_romanisim
else:
    from wpipe.romanisim_util import read_obs_plan, egg_to_romanisim

def register(task):
    _temp = task.mask(source='*', name='start', value=task.name)
    _temp = task.mask(source='*', name='new_isim_target', value='*')

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
    print('event', this_event_id, 'dp', this_dp_id)
    target_id = this_event.options['dp_id']
    target_dp = wp.DataProduct(target_id)
    this_conf = target_dp.config
    param = this_conf.parameters
    obs_plan = read_obs_plan(os.path.join(target_dp.relativepath, target_dp.filename),
                             auxfiles_dir=param['aux_dir'])
    total = len(obs_plan)
    if this_conf.parameters['make_background']:
        ra_idxmin, ra_idxmax = obs_plan['RA'].idxmin(), obs_plan['RA'].idxmax()
        de_idxmin, de_idxmax = obs_plan['DEC'].idxmin(), obs_plan['DEC'].idxmax()
        radec = obs_plan.loc[[ra_idxmin, ra_idxmax, de_idxmin, de_idxmax], ['RA', 'DEC']].to_numpy()
        coo = SkyCoord(radec, frame='icrs', unit='deg')
        # calc centroid and radius
        bg_file = os.path.join(param['background_dir'], 'gg_catalog_2.txt')
        t_bg = egg_to_romanisim(bg_file, ra, dec, radius=radius)
    for i, row in obs_plan.T.to_dict().items():
        my_event = this_job.child_event('new_isim_run', tag=i,
                                        options={'dp_id': dpid, 'to_run': total, 'name': compname,
                                                'submission_type':'scheduler', 'partition': partition,
                                                **row})
        this_job.logprint(''.join(["Firing event ", str(my_event.event_id), "  new_isim_run"]))
        my_event.fire()
