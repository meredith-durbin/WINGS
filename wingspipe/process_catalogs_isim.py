#! /usr/bin/env python

import os
os.environ['CRDS_PATH'] = os.path.join(os.environ['HOME'], 'crds_cache')
os.environ['CRDS_SERVER_URL'] = 'https://roman-crds.stsci.edu/'
import importlib
import json
import shutil
import numpy as np
import pandas as pd
import s3fs
import time
import vaex
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.time import Time
from astropy import units as u

# niche astro
import asdf
import galsim
import hpgeom
import pysiaf

# roman-specific
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


def read_isim_input_catalogs(hplist, catalog_dir, name_template,
                             table_read_kwargs=dict()):
    '''Read HEALPix input catalogs from directory.
    
    Inputs
    ------
    hplist : array-like of int
        List of HEALPix indices
            SOC sims: nest512, Galactic lonlat
            lsstsim: ring256, ICRS lonlat
    catalog_dir : path-like
        Directory or S3 path with catalog files.
    name_template : str
        String for catalog name, to be formatted with healpix index.
        'cat-{}.fits' for SOC input catalogs
        'lsstsim_{}_isim.ecsv' for LSST sims
    table_read_kwargs : dict, default dict()
        Keyword arguments for reading table. Recommended:
        `dict(format='ecsv', engine='pyarrow')` for ecsv
        `dict(format='fits')` for SOC inputs
        
    Returns
    -------
    t : Table
        Stack of all input catalogs found corresponding to HEALPix indices
        in hplist.
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
                tables.append(Table.read(f, **table_read_kwargs))
        else:
            tables.append(Table.read(f, **table_read_kwargs))
    if len(tables) == 0:
        print(f'No catalogs found at {catalog_dir} for HEALPix {hplist}!')
        return Table()
    elif len(tables) == 1:
        t = tables[0]
    else:
        t = vstack(tables)
    return t

def pyananke_to_isim(ds, ab_vega):
    '''Convert vaex dataframe of pyananke-simulated stars to romanisim input.
    
    Inputs
    ------
    ds : vaex DataFrame
        Must have columns ra, dec, roman_<filter>
    ab_vega : dict or pd.Series of floats
        Detector-averaged AB-Vega offsets per WFI filter
    '''
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
    # If input is a string (a file), read it in as an astropy Table
    if type(input_table) == str:
        input_table = Table.read(input_table, format='pandas.csv', sep=' ')
        
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
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
    disk_mags = Table([np.power(10, input_table[f'mag_obs_disk_{i}_AB'].data / -2.5) for i in mag_names],
                           names=mag_names)
    r50_disk = input_table['R50_disk_arcsec']
    ba_disk = input_table['disk_ratio']
    n_disk = np.repeat(1.0, N_gal) # Sersic index of 1 for disk
    disk_table = Table([ra, dec, galtype, n_disk, r50_disk, PA, ba_disk],
                       names=names)
    # Concatenate disk properties and magnitudes into one table
    disk_table = hstack([disk_table, disk_mags])
    if use_bulge: # inject bulge and disk separately, copy  
        bulge_mags = Table([np.power(10, input_table[f'mag_obs_bulge_{i}_AB'].data / -2.5) for i in mag_names],
                           names=mag_names)
        r50_bulge = input_table['R50_bulge_arcsec']
        ba_bulge = input_table['bulge_ratio']
        n_bulge = np.repeat(4.0, N_gal) # Sersic index of 4 for bulge
        bulge_table = Table([ra, dec, galtype, n_bulge, r50_bulge, PA, ba_bulge],
                           names=names)
        # Concatenate bulge properties and magnitudes into one table
        bulge_table = hstack([bulge_table, bulge_mags])
        # Remove bulges fainter than faintest disk
        bulge_table = bulge_table[bulge_table['F087'] > np.min(disk_table['F087'])]
        # Combine bulge and disk tables
        combined_table = vstack([disk_table, bulge_table])
        return combined_table
    else:
        return disk_table
