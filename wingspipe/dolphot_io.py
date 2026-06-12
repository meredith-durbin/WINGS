#!/usr/bin/env python

"""Tools to interact with DOLPHOT ascii outputs.

TODO: 
- add narwhal, polars, astropy table support
- update df_col when new columns added
"""

import glob
import numpy as np
import os
import pandas as pd
import re
import time
import traceback
# import vaex

from pathlib import Path
from typing import Optional

class DolphotOutput:
    def __init__(self, photpath : str | os.PathLike, 
                 paramfile : Optional[str | os.PathLike] = None,
                 refimage : Optional[str | os.PathLike] = None,
                 fakephotfile : Optional[str | os.PathLike] = None,
                 ):
        '''
        Populate output file names based on path to photometry ascii file.
        
        Inputs
        ------
        photpath : path-like
            Path to DOLPHOT ascii photometry table. Assumes all other outputs 
            are in the same directory with the same base name, as DOLPHOT does 
            by default.
        paramfile : path-like or None, default None
            Path to input parameter file used in DOLPHOT run.
        refimage : path-like or None, default None
            Path to astrometric reference image used in DOLPHOT run.
            Currently expects FITS format.
        fakephotfile : path-like or None, default None
            Path to artificial star output file, if ASTs have been run.            
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
        self.paramfile = paramfile
        if (refimage is None) and (paramfile is not None):
            refimage = self._get_refimage_from_param(paramfile)
        self.refimage = refimage
        self.fakephotfile = fakephotfile
        self.column_info = None
        self.phot_table = None
        self.fake_column_info = None
        self.fake_table = None
        self.header_table = None
    
    def get_path(self, attr_name: str) -> os.PathLike | None:
        '''Get absolute path of attribute.
        '''
        full_path = self.basedir.joinpath(getattr(self, attr_name))
        if not full_path.is_file():
            full_path = full_path + '.gz'
            if not full_path.is_file():
                print(f'Cannot locate file: {full_path}')
                return None
        return full_path

    def read_ascii_phot(self, 
                        drop_exposure_cols : bool = False, 
                        dataframe_type : str = 'pandas',
                        fake : bool = False, 
                        trim_input_cols : bool = True, 
                        add_wcs : bool = True, 
                        xcol : str = 'X', 
                        ycol : str = 'Y',
                        add_id : bool = True,
                        id_precision : int = 4,
                        do_culling : bool = True, 
                        param_dict : Optional[dict] = None,):
        '''Read in DOLPHOT ascii photometry file to dataframe.

        Inputs
        ------
        drop_exposure_cols : bool, default False
            Drop all columns with measurements for individual exposures?
        dataframe_type : str, one of ['pandas', 'vaex', 'dask'], default 'pandas'
            Library for reading in and storing photometry table.
        fake : bool, default False
            Whether the photometry file to be read in includes columns 
            corresponding to artificial star test inputs.
        trim_input_cols : bool, default True
            Whether to keep only one AST input column per filter instead of per
            image. Assumes all input magnitudes are the same across all images 
            in a given filter. Only used if `fake` is True.
        add_wcs : bool, default True
            Whether to add RA and Dec columns if reference image is available
        xcol : str, default "X"
            Column with reference image x-coordinate values. Only used if 
            `add_wcs` is True.
        ycol : str, default "Y"
            Column with reference image y-coordinate values. Only used if 
            `add_wcs` is True.
        add_id : bool, default True
            Whether to add ID column based on world coordinates. Only used if 
            `add_wcs` is also True.
        id_precision : int, default 4
            Decimal precision of the last place of sexagesimal notation in the 
            ID string.
        do_culling : bool, default True
            Add columns with ST and GST quality flags?
        param_dict : dict or None, default None
            Dictionary of parameters to be used in culling step. Only used if
            `do_culling` is True.
        '''
        df_col = self._read_colfile(self.get_path('colfile')) if self.column_info is None else self.column_info
        if fake:
            df_col = self._add_fake_input_cols(df_col, trim_input_cols=trim_input_cols)
        if drop_exposure_cols:
            df_col = df_col.query('is_single_image == False')
        ascii_path = self.get_path('fakephotfile') if fake else self.get_path('photfile')
        try:
            df = self._read_photfile(ascii_path, df_col, dataframe_type=dataframe_type)
        except Exception:
            print(f'failed to read {ascii_path}')
            return self
        if add_wcs:
            if (self.refimage is None):
                print('No reference image specified; skipping WCS step')
            else:
                if dataframe_type == 'dask':
                    import dask.dataframe as dd
                ra, dec = self._calc_wcs(df, refimage=self.get_path('refimage'), 
                                         dataframe_type=dataframe_type,
                                         xcol=xcol, ycol=ycol)
                if dataframe_type == 'vaex':
                    col_order = df.get_column_names(virtual=True, hidden=False)
                elif dataframe_type in ['pandas', 'dask']:
                    col_order = df.columns.tolist()
                y_idx = col_order.index(ycol) + 1
                col_reorder = col_order[:y_idx] + ['RA', 'DEC'] + col_order[y_idx:]
                if add_id:
                    id_col = self._make_id_from_radec(ra, dec, precision=id_precision)
                    col_reorder = ['ID'] + col_reorder
                    if dataframe_type == 'vaex':
                        df.add_column('ID', id_col, dtype=id_col.dtype)
                    elif dataframe_type == 'pandas':
                        df.insert(0, 'ID', id_col)
                    elif dataframe_type == 'dask':
                        print('ID assignment not yet implemented for dask dataframes')
                        # df['ID'] = pd.Series(id_col, index=df.index).\
                        #     pipe(dd.from_pandas, npartitions=df.npartitions)
                if dataframe_type == 'vaex':
                    df.add_column('RA', ra, dtype=ra.dtype)
                    df.add_column('DEC', dec, dtype=dec.dtype)
                elif dataframe_type == 'pandas':
                    df.insert(y_idx, 'DEC', dec)
                    df.insert(y_idx, 'RA', ra)
                elif dataframe_type == 'dask':
                    print('WCS assignment not yet implemented for dask dataframes')
                    # df['RA'] = pd.Series(ra, index=df.index).\
                    #     pipe(dd.from_pandas, npartitions=df.npartitions)
                    # df['DEC'] = pd.Series(dec, index=df.index).\
                    #     pipe(dd.from_pandas, npartitions=df.npartitions)
                df = df[col_reorder]
        if do_culling:
            df = self._cull_photometry(df, param_dict=param_dict, 
                                       dataframe_type=dataframe_type)
        if fake:
            self.fake_column_info = df_col
            self.fake_table = df
        else:
            # self.column_info = df_col
            self.phot_table = df
        return self
    
    @staticmethod
    def _get_refimage_from_param(paramfile):
        # needs work!!!
        refimage = None
        with open(paramfile, 'r') as f:
            for line in f:
                if line.startswith('img0_file'):
                    refimage = line.split('=')[-1].strip()
                    break
        if refimage is None:
            print(f'No reference image found in {paramfile}')
        else:
            refimage = glob.glob(refimage + '.*')[0]
        return refimage
    
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
        is_single_filter = df_col['description'].str.split(re.escape(' (')).str[0].\
            str.contains(re.escape(', '))
        single_image_regex = r'\([A-Z0-9]{1,9}\_[fF][0-9]{3,4}[a-zA-Z][0-9]?, [0-9]{1,5}\.[0-9]{1,4} sec\)$'
        is_single_image = df_col['description'].str.contains(single_image_regex, regex=True)
        df_col['is_single_filter'] = is_single_filter
        df_col['is_single_image'] = is_single_image
        df_single = df_col.loc[is_single_filter]
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
        uint8_cols = df_col.filter(regex='^(EXT|CHIP|MAJAX|OBJTYPE|PASS)$|(.*_FLAG$)', axis=0).index
        df_col.loc[uint8_cols, 'dtype'] = 'uint8'
        # if 'PASS' in df_col.index:
        #     df_col.loc['PASS', 'dtype'] = 'uint8'
        # add null values
        df_col['na_values'] = ''
        for k, v in na_values.items():
            df_col.loc[df_col.filter(regex=f'.*\\_{k}$', axis=0).index, 'na_values'] = v
        # add units
        df_col['unit'] = ''
        df_col.loc[['X', 'Y'], 'unit'] = 'pixel'
        mag_cols = df_col.filter(regex='.*_(MAG|VEGA|TRANS|CROWD|ERR)$', axis=0).index
        df_col.loc[mag_cols, 'unit'] = 'mag'
        count_cols = df_col.filter(regex='.*_(COUNT|SKY)$', axis=0).index
        df_col.loc[count_cols, 'unit'] = 'count'
        df_col['in_ascii'] = True
        return df_col
    
    @staticmethod
    def _read_photfile(photfile : str | os.PathLike, 
                       df_col : pd.DataFrame, 
                       dataframe_type : str = 'pandas',
                       ):
        '''Read in DOLPHOT ascii photometry file to vaex dataframe.

        Inputs
        ------
        photfile : str or pathlike
            Photometry ascii file
        df_col : pandas.DataFrame
            Table of column information, such as from DolphotOutput.read_colfile
        dataframe_type : str, one of ['vaex', 'pandas', 'dask'], default 'vaex'
            Library to use for reading in ascii file.
            
        Returns
        -------
        df : dataframe
            Photometry table.
        '''
        csv_reader_kwargs = dict(sep=r'\s+',
                                 names=df_col.index.tolist(), 
                                 usecols=df_col['num'].tolist(), 
                                 na_values=df_col['na_values'].to_dict(), 
                                 keep_default_na=False,
                                 dtype=df_col['dtype'].to_dict(),
                                 low_memory=True,
                                 float_precision='round_trip',
                                 )
        if dataframe_type == 'vaex':
            import vaex
            df = vaex.from_csv(photfile, **csv_reader_kwargs,
                               copy_index=False, 
                               )
        elif dataframe_type == 'dask':
            import dask.dataframe as dd
            df = dd.read_csv(photfile, **csv_reader_kwargs)
        elif dataframe_type == 'pandas':
            df = pd.read_csv(photfile, **csv_reader_kwargs)
        return df
    
    @staticmethod
    def _add_fake_input_cols(df_col : pd.DataFrame, 
                             trim_input_cols : bool = True,
                             ) -> pd.DataFrame:
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
            df_col.loc[is_mag_in, 'is_single_image'] = False
            df_col = df_col.drop_duplicates(subset=['name'], keep='first')
        return df_col.set_index('name')
    
    @staticmethod
    def _calc_wcs(df, 
                  refimage : str | os.PathLike, 
                  dataframe_type : str = 'pandas',
                  xcol : str = 'X', 
                  ycol : str = 'Y',
                  ):
        '''Convert X and Y columns to world coordinates with refimage WCS.
        
        Inputs
        ------
        df : dataframe
            Photometry table
        refimage : str or path-like object
            Path to reference image FITS file with WCS specification
        dataframe_type : str, one of ['pandas', 'vaex', 'dask'], default 'pandas'
            Library for reading in and storing photometry table.
        xcol : str, default "X"
            Column with reference image x-coordinate values.
        ycol : str, default "Y"
            Column with reference image y-coordinate values.

        Returns
        -------
        ra, dec : tuple of arrays
            1D arrays of RA and Dec values in degrees.
        '''
        from astropy.io import fits
        from astropy.wcs import WCS
        # need more robust way of getting right WCS here
        with fits.open(refimage, mode='readonly', memmap=False) as f:
            if ('chip' in f.filename()) or ('wcs' in f.filename()):
                w = WCS(f[0].header, fobj=f)
            else:
                w = WCS(f[1].header, fobj=f)
        # dolphot pixel coordinates are half a pixel off from every other convention
        if dataframe_type == 'vaex':
            x, y = df.evaluate([f'{xcol}-0.5', f'{ycol}-0.5'], array_type='numpy')
        elif dataframe_type == 'pandas':
            x, y = df.eval(f'{xcol}-0.5').to_numpy(), df.eval(f'{ycol}-0.5').to_numpy()
        elif dataframe_type == 'dask':
            x, y = df.eval(f'{xcol}-0.5').compute(), df.eval(f'{ycol}-0.5').compute()
        ra, dec = w.all_pix2world(x, y, 0, ra_dec_order=True)
        return ra, dec
    
    @staticmethod
    def _make_id_from_radec(ra : np.typing.ArrayLike, 
                            dec : np.typing.ArrayLike,
                            precision : int = 4):
        '''Make string ID from RA and Dec arrays.
        
        Inputs
        ------
        ra : array-like
            Array of right ascension values in degrees.
        dec : array-like
            Array of declination values in degrees.
        precision : int, default 4
            Decimal precision of the last place of sexagesimal notation.
            
        Returns
        -------
        id_str : array-like of strings
            Array of source ID strings, of the format 
            'J<HHMMSS.SSSS+DDMMSS.SSSS>'
        '''
        from astropy.coordinates import SkyCoord
        coo = SkyCoord(ra, dec, unit='deg', frame='icrs')
        ra_str = coo.ra.to_string(unit='hourangle', decimal=False, sep='', 
                                  precision=precision, pad=True)
        de_str = coo.dec.to_string(unit='deg', decimal=False, sep='', 
                                   precision=precision, pad=True, alwayssign=True)
        id_str = 'J' + pd.Series(ra_str).str.cat(pd.Series(de_str), sep='')
        return id_str.to_numpy().astype('str')
    
    @staticmethod
    def _cull_photometry(df, 
                         param_dict : Optional[dict] = None,
                         dataframe_type : str = 'pandas', 
                         verbose : bool = False
                         ):
        """Make ST ("star") and GST ("good star") selections on photometry catalog.

        Inputs
        ------
        df : dataframe
            Dataframe of photometry or ASTs.
        param_dict : dict or None, default None
            Dictionary of custom culling parameters.
        verbose : bool, default False
            Whether to print messages about numbers of stars passing quality 
            cuts per column.
        
        Returns
        -------
        df : dataframe
            Input dataframe with added ST and GST columns.
        """
        if dataframe_type == 'vaex':
            mag_cols = pd.Series(df.get_column_names(regex='.*_(VEGA|MAG)$'))
            pass_col = df.get_column_names(regex='^PASS$')
        elif dataframe_type == 'dask':
            mag_cols = [c for c in df.columns if c.endswith('VEGA') or c.endswith('MAG')]
            pass_col = [c for c in df.columns if c == 'PASS']
        else:
            mag_cols = df.filter(regex='.*_(VEGA|MAG)$').columns
            pass_col = df.filter(regex='^PASS$')
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
        if 'PASS' in pass_col:
            univ_keys += ['passcut']
        for col in mag_cols:
            prefix = re.split(r'_(VEGA|MAG)$', col)[0]
            if (re.match('^WFC3_F[2-9]', prefix) is not None) or (re.match('^i[a-z0-9]{8}_[fF][2-9]', prefix) is not None):
                detector = 'uvis'
            elif (re.match('^WFC3_F[01]', prefix) is not None) or (re.match('^i[a-z0-9]{8}_[fF][01]', prefix) is not None):
                detector = 'ir'
            elif ('ACS' in prefix) or (re.match('^j[a-z0-9]{8}_', prefix) is not None):
                detector = 'wfc'
            elif ('NIRCAM' in prefix) or ('NIRISS' in prefix) or (re.match('^jw[0-9]{11}_[0-9]{5}_[0-9]{5}_n', prefix) is not None):
                detector = 'nircam'
            elif ('ROMAN' in prefix) or (re.match('^r[0-9]{19}_[0-9]{4}_wfi[01][0-9]_f[0-9]{3}_cal', prefix) is not None):
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
            st_str = '(OBJTYPE < {1:d}) & ({0}_SNR > {2:.3f}) & ({0}_FLAG < {3:d}) & ({0}_ROUND < {4:.3f}) & ({0}_SHARP < {5:.3f})'.\
                format(prefix, cuts['objcut'], cuts['snrcut'], cuts['flagcut'], cuts['roundcut'], cuts[f'{detector}_sharp'])
            if 'passcut' in cuts.keys():
                st_str += ' & (PASS < {0:d})'.format(cuts['passcut'])
            gst_str = st_str + ' & ({0}_CROWD < {1:.3f})'.format(prefix, cuts[f'{detector}_crowd'])
            if dataframe_type == 'vaex':
                df[f'{prefix}_ST'] = df.func.where(df[st_str], True, False, dtype='bool')
                df[f'{prefix}_GST'] = df.func.where(df[gst_str], True, False, dtype='bool')
            elif dataframe_type in ['pandas', 'dask']:
                df = df.eval(f'{prefix}_ST = {st_str}')
                df = df.eval(f'{prefix}_GST = {gst_str}')
                # df[f'{prefix}_ST'] = df.eval(st_str).astype('bool')
                # df[f'{prefix}_GST'] = df.eval(gst_str).astype('bool')
            if verbose:
                n_st = df[f'{prefix}_ST'].sum()
                n_gst = df[f'{prefix}_GST'].sum()
                df_len = df.length() if dataframe_type == 'vaex' else len(df)
                print(f'Found {n_st} out of {df_len} stars meeting ST criteria in {prefix}')
                print(f'Found {n_gst} out of {df_len} stars meeting GST criteria in {prefix}')
        return df