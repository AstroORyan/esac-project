'''
Author: David O'Ryan
Date: 29/04/2022
Quick script to create the filepath to ACS/WFC Observation on DataLabs from the HST or HLA obs_collection.

Requires Astroquery and assumes that the mounted data volume names are hub_hstdata_{instrument_designation} for HST data or hap_hstdata_{instrument_designation} for HAP observations. 

Note: for instrument_designation = j for ACS.

Expects input of a SkyCoord to call Astroquery, the filter, the wanted filetype and the specific observation id wanted (if calling for HLA only).
'''
## Imports
import sys
import glob

from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.mast import Observations

## Functions
def create_url(
    coords: SkyCoord,
    req_filter: str,
    spec_obsid = None,
    instrument = 'ACS/WFC',
    collection = ['HST'],
    filetype = 'drc'
):
    obs_table = Observations.query_criteria(
        coordinates = coords,
        radius = 20 * u.arcsec,
        dataproduct_type = 'image',
        instrument_name = instrument,
        obs_collection = collection,
        calib_level = 3,
        filters = [req_filter]
    )

    if len(obs_table) == 0:
        return None

    obs_table.sort(['t_exptime'])
    obs_table.reverse()

    obs = obs_table[0]
    obsid = obs['obs_id']

    if obs['instrument_name'] == 'ACS/WFC':
        desig = 'j'
        split_instr = obs['instrument_name'].lower().split('/')
        tmp_1 = split_instr[0] 
        tmp_2 = split_instr[1]
        full_instr = f'{tmp_1}_{tmp_2}'
    else:
        print('WARNING: not implemented script for other instruments yet. Exiting for safety')
        sys.exit()

    if len(obsid) >= 15:
        id_list = obsid.split('_')
        telesc = id_list[0]
        proj_id = id_list[1]
        sub_id = id_list[2]
        instr = id_list[3]

        if spec_obsid is None:
            file_path = f'/media/home/data/hap_hstdata_{desig}/{proj_id}/{sub_id}/{instr}/{telesc}_{proj_id}_{sub_id}_{full_instr}_{req_filter}_*_{filetype}.fits.gz'
            test = glob.glob(file_path)
            if len(test) == 0.0:
                print(f'WARNING: File path construction has failed for {obsid}. Stopping for safety, and should investigate.')
            file_path = test[0]
        else:
            file_path = f'/media/home/data/hap_hstdata_{desig}/{proj_id}/{sub_id}/{instr}/{telesc}_{proj_id}_{sub_id}_{full_instr}_{req_filter}_{spec_obsid}_{filetype}.fits.gz'
    
    elif len(obsid) < 15:
        obs_program = obsid[1:4]
        obs_set_id = obsid[4:6]

        file_path = f'/media/home/data/hub_hstdata_{desig}/{desig}/{obs_program}/{obs_set_id}/{obsid}_{filetype}.fits.gz'

    test = glob.glob(file_path)
    if len(test) == 0.0:
        print(f'WARNING: File path construction has failed for {obsid}. Stopping for safety, and should investigate.')

    return file_path


