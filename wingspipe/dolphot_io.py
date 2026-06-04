#!/usr/bin/env python

"""Tools to interact with DOLPHOT ascii outputs.
"""

import argparse
import glob
import numpy as np
import os
import pandas as pd
import re
import time
import traceback
import vaex

from pathlib import Path
from typing import Optional

class DolphotOutput:
    def __init__(self, photpath : str | os.PathLike):
        '''
        Populate output file names based on path to photometry ascii file.
        '''
        absphotpath = Path(photpath).absolute()
        self.basename = absphotpath.stem
        self.basedir = absphotpath.parent
        self.photfile = absphotpath.name
        self.alignfile = f'{self.photfile}.align'
        self.apcorfile = f'{self.photfile}.apcor'
        self.colfile = f'{self.photfile}.columns'
        self.infofile = f'{self.photfile}.info'
        self.psfsfile = f'{self.photfile}.psfs'
        self.warnfile = f'{self.photfile}.warnings'
        
    def get_path(self, attr_name: str) -> os.PathLike | None:
        full_path = self.basedir.joinpath(getattr(self, attr_name))
        if not full_path.is_file():
            full_path = full_path + '.gz'
            if not full_path.is_file():
                print(f'Cannot locate file: {full_path}')
                return None
        return full_path
    
    @staticmethod
    def _read_colfile(colfile : str | os.PathLike) -> pd.DataFrame:
        """Construct a dataframe of Dolphot output column information.

        Inputs
        ------
        colfile : str, path object or file-like object
            Path to DOLPHOT '.columns' file; see pandas.read_csv

        Returns
        -------
        df_col : pandas.DataFrame
            Table of column info, including names, descriptions, ordering, 
            filter names, and null values.
        """
        
        # Mappings of column descriptions to output names for single-field columns
        replace_single = {
            '(Total|Measured) counts'           : 'COUNT',
            '(Total|Measured) sky level'        : 'SKY',
            'Normalized count rate uncertainty' : 'RATERR',
            'Normalized count rate'             : 'RATE',
            'Instrumental VEGAMAG magnitude'    : 'VEGA',
            'Instrumental magnitude'            : 'MAG',
            'Transformed UBVRI magnitude'       : 'TRANS',
            'Magnitude uncertainty'             : 'ERR',
            'Photometry quality flag'           : 'FLAG',
            'PSF FWHM'                          : 'FWHM',
            'PSF eccentricity'                  : 'PSFE',
            'PSF a parameter'                   : 'PSFA',
            'PSF b parameter'                   : 'PSFB',
            'PSF c parameter'                   : 'PSFC',
        }

        # Same for global columns
        replace_global = {
            'Object'          : '',
            'Signal-to-noise' : 'SNR',
            '(ness)|(ing)'    : '',
            'Pass Detected'   : 'PASS',
        }

        # Mappings of column name suffixes to NA values
        na_values = {
            'RATERR' : '9999.000',
            'VEGA'   : '99.999',
            'MAG'    : '99.999',
            'TRANS'  : '99.999',
            'ERR'    : '9.999',
            'RATE'   : '0.00e+00',
        }

        # read column file into dataframe with index (column number) and description
        df_col = pd.read_csv(colfile, names=['num_orig', 'description'],
                             sep=re.escape('. '), engine='python').rename_axis('num')
        # select for single-exposure columns with descriptions formatted as
        # "<quantity>, <filename> (<filter>, <exptime>)"
        is_single = df_col['description'].str.split(re.escape(' (')).str[0].\
            str.contains(re.escape(', '))
        df_single = df_col.loc[is_single]
        # all other columns (XY position, object type, combined magnitudes, etc)
        df_global = df_col.drop(df_single.index)

        # parse global column descriptions and convert to names
        names_global = df_global['description'].str.replace('Object', '').\
            str.replace('Extension', 'EXT', regex=True).\
            str.replace('Direction', 'MAJAX', regex=True).\
            str.replace('type', 'OBJTYPE', regex=True).\
            str.strip().str.split().str[0]
        
        # parse single-exposure descriptions and convert to columns names
        single_split = df_single['description'].str.split(re.escape(', '))
        names_single = single_split.str[0]
        for k, v in replace_single.items():
            names_single = names_single.replace(re.compile(k), v)
        names_concat = pd.concat([names_global, names_single]).reindex_like(df_col)
        for k, v in replace_global.items():
            names_concat = names_concat.replace(re.compile(k), v)
        prefix_global = pd.Series('', index=names_global.index)
        prefix_single = single_split.str[1].str.split().str[0]
        prefix_concat = pd.concat([prefix_global, prefix_single]).reindex_like(df_col)
        # names = prefix_concat.str.cat(names_concat.str.upper(), sep='_').\
        #             str.replace('(^\_)|(\_$)', '', regex=True)
        # df_col.loc[:, 'name'] = names
        df_col.loc[:, 'prefix'] = prefix_concat.str.replace(r'.', r'_', regex=False).str.replace(r'-', r'_', regex=False)
        df_col.loc[:, 'suffix'] = names_concat.str.upper()
        df_col['name'] = df_col['prefix'].str.cat(df_col['suffix'], sep='_').str.replace('(^\\_)|(\\_$)', '', regex=True)
        # NB: won't parse filter names for really old dolphot versions that didn't prepend instruments to filter names
        df_col = df_col.join(df_col['description'].str.extract(r'(?P<filt>[A-Z0-9]{1,9}\_[fF][0-9]{3,4}[a-zA-Z][0-9]?)'))
        df_col = df_col.reset_index().set_index('name')
        df_col['dtype'] = 'float32'
        uint8_cols = df_col.filter(regex='^(EXT|CHIP|MAJAX|OBJTYPE)$|(.*_FLAG$)', axis=0).index
        df_col.loc[uint8_cols, 'dtype'] = 'uint8'
        if 'PASS' in df_col.index:
            df_col.loc['PASS', 'dtype'] = 'uint32' # some of the passes are huge numbers for some reason
        df_col['na_values'] = ''
        for k, v in na_values.items():
            df_col.loc[df_col.filter(regex=f'.*\\_{k}$', axis=0).index, 'na_values'] = v
        df_col['in_orig_ascii'] = True
        return df_col
    
    @staticmethod
    def _read_photfile(photfile : str | os.PathLike, df_col : pd.DataFrame, 
                       keep_exposure_cols : bool = True, 
                       refimage : Optional[str | os.PathLike] = None, 
                       xcol : str = 'X', ycol : str = 'Y',
                       do_culling : bool = True, param_dict : Optional[dict] = None,
                       ) -> vaex.dataframe.DataFrame:
        '''Read in DOLPHOT ascii photometry file to vaex dataframe.

        Inputs
        ------
        photfile : str or pathlike
            Photometry ascii file
        df_col : pandas.DataFrame
            Table of column information, such as from DolphotOutput.read_colfile
        keep_exposure_cols : bool, default False
            Keep all columns with measurements for individual exposures?
        refimage : str or path-like object, optional
            Optional path to reference image FITS file with WCS specification
        xcol : str, default "X"
            Column with reference image x-coordinate values. Only used if 
            `refimage` is not `None`.
        ycol : str, default "Y"
            Column with reference image y-coordinate values. Only used if 
            `refimage` is not `None`.
        do_culling : bool, default True
            Add columns with ST and GST quality flags?
        param_dict : dict or None, optional
            Dictionary of parameters to be used in culling step. Only used if
            `do_culling` is `True`.
            
        Returns
        -------
        ds : vaex.dataframe.DataFrame
            Vaex dataframe of photometry.
        '''
        if not keep_exposure_cols:
            df_col = df_col[~df_col.index.str.contains('chip[0-9]')]
        ds = vaex.from_csv(photfile, copy_index=False, sep=r'\s+',
                           names=df_col.index.tolist(), usecols=df_col['num'].tolist(),
                           na_values=df_col['na_values'].to_dict(),
                           dtype=df_col['dtype'].to_dict())
        if refimage is not None:
            ds = DolphotOutput._add_wcs(ds, refimage=refimage, xcol=xcol, ycol=ycol)
        if do_culling:
            ds = DolphotOutput._cull_photometry(ds, param_dict=param_dict)
        return ds
    
    @staticmethod
    def _add_fake_input_cols(df_col : pd.DataFrame, trim_input_cols : bool = True):
        '''Add columns corresponding to AST inputs to column dataframe.
        
        Inputs
        ------
        df_col : pandas.DataFrame
            Table of column information, such as from DolphotOutput.read_colfile
        trim_input_cols : bool, default True
            Whether to keep only one input column per filter instead of per 
            image. Assumes all input magnitudes are the same across all images 
            in a given filter.
            
        Returns
        -------
        df_col : pandas.DataFrame
            Table of column information with AST columns added.
        '''
        input_cols = pd.concat([df_col.loc[['EXT', 'CHIP', 'X', 'Y']],
                                df_col.filter(regex='.*chip[0-9]*\\_(COUNT|VEGA|MAG)$', axis=0)],
                            ignore_index=False)
        input_cols.index = input_cols.index + '_IN'
        input_cols['suffix'] = input_cols['suffix'] + '_IN'
        input_cols['description'] = input_cols['description'].str.replace('(Instrumental|Measured|Object)', 'Input', regex=True)
        input_cols['num'] = pd.Series(pd.RangeIndex(start=0, stop=len(input_cols)), index=input_cols.index)
        df_col = pd.concat([input_cols, df_col.eval(f'num = num + {len(input_cols)}')], 
                            ignore_index=False).reset_index()
        if trim_input_cols:
            is_mag_in = df_col['suffix'].eq('VEGA_IN')
            if is_mag_in.sum() > 0:
                df_col.loc[is_mag_in, 'name'] = df_col['filt'].str.upper().str.cat(df_col['suffix'], sep='_').loc[is_mag_in]
            else:
                is_mag_in = df_col['suffix'].eq('MAG_IN')
                df_col.loc[is_mag_in, 'name'] = df_col['prefix'].str.cat(df_col['suffix'], sep='_').loc[is_mag_in]
            df_col = df_col.drop_duplicates(subset=['name'], keep='first')
        # df_col = df_col[~df_col.index.duplicated(keep='first')]
        return df_col.set_index('name')
    
    @staticmethod
    def _add_wcs(ds : vaex.dataframe.DataFrame, refimage : str | os.PathLike, 
                 xcol : str = 'X', ycol : str = 'Y') -> vaex.dataframe.DataFrame:
        '''Convert X and Y columns to world coordinates with refimage WCS.
        
        Inputs
        ------
        ds : vaex.dataframe.DataFrame
            Photometry table
        refimage : str or path-like object
            Path to reference image FITS file with WCS specification
        xcol : str, default "X"
            Column with reference image x-coordinate values.
        ycol : str, default "Y"
            Column with reference image y-coordinate values.

        Returns
        -------
        ds : vaex.dataframe.DataFrame
            Photometry table with RA and Dec columns inserted after `ycol`.
        '''
        from astropy.io import fits
        from astropy.wcs import WCS
        # need more robust way of getting right WCS here
        with fits.open(refimage, mode='readonly', memmap=False) as f:
            if ('chip' in refimage) or ('wcs' in refimage):
                w = WCS(f[0].header, fobj=f)
            else:
                w = WCS(f[1].header, fobj=f)
        col_order = ds.get_column_names(virtual=True, hidden=False)
        y_idx = col_order.index(ycol)+1
        col_reorder = col_order[:y_idx] + ['RA', 'DEC'] + col_order[y_idx:]
        # dolphot pixel coordinates are half a pixel off from everything else
        ra, dec = w.all_pix2world(*ds.evaluate([f'{xcol}-0.5', f'{ycol}-0.5'], array_type='numpy'), 0, ra_dec_order=True)
        ds.add_column('RA', ra)
        ds.add_column('DEC', dec)
        return ds[col_reorder]
    
    @staticmethod
    def _make_id_from_radec(ra : np.typing.ArrayLike, dec : np.typing.ArrayLike,
                            precision : int = 5):
        '''Make string ID from RA and Dec arrays.
        
        Inputs
        ------
        ra : array-like
            Array of RA values in degrees.
        dec : array-like
            Array of Dec values in degrees.
        precision : int, default 5
            Decimal precision of the last place of sexagesimal notation.
        '''
        from astropy.coordinates import SkyCoord
        coo = SkyCoord(ra, dec, unit='deg', frame='icrs')
        ra_str = coo.ra.to_string(unit=u.hourangle, decimal=False, sep='', precision=precision, pad=True)
        de_str = coo.dec.to_string(unit=u.deg, decimal=False, sep='', precision=precision, pad=True, alwayssign=True)
        id_str = 'J' + pd.Series(ra_str).str.cat(pd.Series(de_str), sep='')
        return id_str.to_numpy().astype('str')
    
    @staticmethod
    def _cull_photometry(ds : vaex.dataframe.DataFrame, param_dict : Optional[dict] = None,
                         verbose : bool = False) -> vaex.dataframe.DataFrame:
        """Make ST ("star") and GST ("good star") selections on photometry catalog.

        Inputs
        ------
        ds : vaex.dataframe.DataFrame
            Vaex dataframe of photometry or ASTs.
        param_dict : dict or None, default None
            Dictionary of custom culling parameters.
        verbose : bool, default False
            Whether to print messages about numbers of stars passing quality 
            cuts per column.
        
        Returns
        -------
        ds : vaex.dataframe.DataFrame
            Input dataframe with added ST and GST columns.
        """
        mag_cols = pd.Series(ds.get_column_names(regex='.*_(VEGA|MAG)$'))
        # make initial selections by filter
        default_cuts = {'objcut': 3, 'snrcut': 4.0, 'flagcut': 4, 
                        'passcut': 3, 'roundcut': 0.5,
                        'default_sharp': 0.15, 'default_crowd': 1.5,
                        'ir_sharp': 0.15, 'ir_crowd': 2.25,
                        'uvis_sharp': 0.15, 'uvis_crowd': 1.3,
                        'wfc_sharp': 0.2, 'wfc_crowd':2.25,
                        'nircam_sharp': 0.01, 'nircam_crowd': 0.5,
                        'roman_sharp': 0.15, 'roman_crowd': 0.5,
                        }
        univ_keys = ['objcut', 'snrcut', 'flagcut', 'roundcut']
        if 'PASS' in ds.get_column_names():
            univ_keys += ['passcut']
        for col in mag_cols:
            prefix = re.split(r'_(VEGA|MAG)$', col)[0]
            if (re.match('^WFC3_F[2-9]', prefix) is not None) or (re.match('^i[a-z0-9]{8}_[fF][2-9]', prefix) is not None):
                detector = 'uvis'
            elif (re.match('^WFC3_F[01]', prefix) is not None) or (re.match('^i[a-z0-9]{8}_[fF][01]', prefix) is not None):
                detector = 'ir'
            elif ('ACS' in prefix) or (re.match('^j[a-z0-9]{8}_', prefix) is not None):
                detector = 'wfc'
            elif ('NIRCAM' in prefix) or ('NIRISS' in prefix) or (re.match('^jw[0-9]{11}_[0-9]{5}_[0-9]{5}_nrc', prefix) is not None):
                detector = 'nircam'
            elif ('ROMAN' in prefix) or (re.match('^r[0-9]{19}_[0-9]{4}_wfi[01][0-9]_f[0-9]{3}_cal') is not None):
                detector = 'roman'
            else:
                detector = 'default'
                print(f'Unable to parse detector name from {col}; using default culling criteria')
            config_keys = univ_keys + [f'{detector}_sharp', f'{detector}_crowd']
            if param_dict is None:
                cuts = {k: default_cuts[k] for k in config_keys}
            else:
                cuts = {}
                for k in config_keys:
                    if k not in param_dict.keys():
                        print(f'Parameter key {k} not found; setting to {default_cuts[k]}')
                        cuts[k] = default_cuts[k]
                    else:
                        cuts[k] = param_dict[k]
            st_str = '(OBJTYPE < {1}) & ({0}_SNR > {2}) & ({0}_FLAG < {3}) & ({0}_ROUND < {4}) & ({0}_SHARP < {5})'.\
                format(prefix, cuts['objcut'], cuts['snrcut'], cuts['flagcut'], cuts['roundcut'], cuts[f'{detector}_sharp'])
            if 'passcut' in cuts.keys():
                st_str += ' & (PASS < {0})'.format(cuts['passcut'])
            gst_str = st_str + ' & ({0}_CROWD < {1})'.format(prefix, cuts[f'{detector}_crowd'])
            ds[f'{prefix}_ST'] = ds.func.where(ds[st_str], True, False, dtype='bool')
            ds[f'{prefix}_GST'] = ds.func.where(ds[gst_str], True, False, dtype='bool')
            if verbose:
                n_st = ds[f'{prefix}_ST'].sum()
                print(f'Found {n_st} out of {ds.length()} stars meeting ST criteria in {prefix}')
                n_gst = ds[f'{prefix}_GST'].sum()
                print(f'Found {n_gst} out of {ds.length()} stars meeting GST criteria in {prefix}')
        return ds