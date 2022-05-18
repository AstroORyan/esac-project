'''
Author: David O'Ryan
Date: 09/05/2022

These set of functions will take in all of the _drc.fits files from the Hubble Legacy Archive and begin to loop through them to create cutouts to go into Zoobot. This will be one of two ways that this can be done. 

Will use glob to find all the files we can use, will then use the HSC to identify all of the sources within those files and then create the cutouts. These cutouts will then be put into Zoobot to create a predictions dataframe,
and save with the fits files name in it. This prediction file will also contain the right ascension, declination of each source. 

Let's go...
'''
## Imports
import argparse
import logging
import argparse
import glob
import sys
import os
from tqdm import tqdm
import gc
import json
import numpy as np

import zoobot_utils
import getting_hsc_coords


## Functions
def test_name(filename):
    name_list = filename.split('_')
    if len(name_list[6]) != 6:
        return True
    else:
        return False

def get_paths(inst, filt, filetype, debug_flag, part):
    datalabs_folder = '/media/home/data'
    if inst == 'acs':
        volume = 'hap_hstdata_j'
    
    if debug_flag:
        proj_ids_path = glob.glob(f'{datalabs_folder}/{volume}/*')
        for i in proj_ids_path:
            paths_tmp = glob.glob(f'{i}/*/{inst}/*{filt}*_{filetype}.fits.gz')
            if paths_tmp:
                break
        paths = paths_tmp[:100]
    elif part == any(range(1,100)):
        with open(f'/media/home/my_workspace/files_list/file_list_part_{part}.json', 'r') as f:
            paths_dict = json.load(f)
        keys_list = list(paths_dict.keys())
        paths = []
        for i in keys_list:
            paths.append(paths_dict[i][0])
    elif part == 'all':
        with open(f'/media/home/my_workspace/files_list/full_list.json', 'r') as f:
            paths_dict = json.load(f)
        keys_list = list(paths_dict.keys())
        paths = []
        for i in keys_list:
            paths.append(paths_dict[i][0])
    else:
        logging.info('Conflicting instructions about debug mode and no file parts provided. Exiting and please check argument inputs.')
        sys.exit()

    return paths

def batch(data, n=50):
    n_batches = int(np.ceil(len(data)/n))
    counter = 0
    batches = []
    for i in range(n_batches):
        if i < n_batches - 1:
            batches.append(data[counter: counter + n])
            counter += n
        else:
            batches.append(data[counter:])
    return batches

## Main Function
def main(debug_flag : bool, inst: str, filt: str, filetype: str, part):
    save_folder = '/media/home/my_workspace/cutouts/tmp'
    if debug_flag:
        logging.info('Debugging flag set to true. Will only predict on a maximum of 5 files.')
    
    logging.info(f'Searching the archive for {filetype} files in the {filt} filter from the instrument: {inst}.')
    logging.info(f'Loading Zoobot Model from 2022-05-05. Has a best cutoff of 0.4817 and an accuracy of 0.8923.')

    all_paths = get_paths(inst, filt, filetype, debug_flag, part)

    if debug_flag:
        batches = [all_paths]
    else:
        batches = batch(all_paths)
    
    for i in range(len(batches)):
        logging.info(f'Beginning batch {i} of {len(batches)}.')
        for j in tqdm(batches[i]):
            filename = os.path.basename(j).replace('.fits.gz','')
            skip_flag = test_name(filename)

            if skip_flag:
                # logging.info(f'Bad file found in {filename}. Skipping')
                # gc.collect()
                continue

            if os.path.exists(f'{save_folder}/{os.path.basename(j).replace(".fits.gz","")}/pred_{os.path.basename(j).replace(".fits.gz","")}.csv'):
                continue
            data, wcs, coords_dict = getting_hsc_coords.get_coords(j)

            if data is None:
                continue

            skip_flag = getting_hsc_coords.test_func(data, wcs, coords_dict)

            if skip_flag:
                continue
            
            if os.path.exists(f'{save_folder}/{filename}'):
                pass
            else:
                os.mkdir(f'{save_folder}/{filename}')

            logging.info(f'Creating a maximum of {len(coords_dict)} cutouts. Standby...')
            for k in tqdm(list(coords_dict.keys())):
                coord = coords_dict[k]
                getting_hsc_coords.cutout_creation(coord, k, data, wcs, save_folder, filename)
            logging.info('Complete.')

            del data, wcs, coords_dict, coord

        model = zoobot_utils.load_model()
        total_sources = 0
        total_interactors = 0

        logging.info('Beginning Predictions on batch...')
        for k in tqdm(batches[i]):
            filename = os.path.basename(k).replace('.fits.gz','')
            #logging.info('Beginning predictions...')

            if os.path.exists(f'{save_folder}/{filename}'):
                pass
            else:
                continue

            if os.path.exists(f'{save_folder}/{filename}/pred_{filename}.csv'):
                logging.info('Predictions previously made. Skipping...')
                continue
            else:
                pass

            manifest_df = zoobot_utils.get_manifest(save_folder, filename)
            predictions = zoobot_utils.get_predictions(manifest_df, model, save_folder, filename)
            #logging.info('Complete.')

            file = os.path.basename(k)
            n_sources = len(predictions)
            n_interactors = predictions.binary_prediction.value_counts()[1]
            n_non_interactors = predictions.binary_prediction.value_counts()[0]

            total_interactors += n_interactors
            total_sources += n_sources

            logging.info(f'After looking through {file} {n_sources} sources were found. Of these, some {n_interactors} were interacting galaxies, while {n_non_interactors} were not. This was in batch {i}.')

            del manifest_df, predictions
        
        logging.info(f'Of {total_sources} scanned in batch {i}; {total_interactors} interacting galaxies were found. Is this number too high or too small? Note, duplicates have NOT been accounted for!')
        
        del model

        gc.collect()

## Initilization
if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description='Conducting predictions on the entire HSC.')
    parser.add_argument('--debug', dest='debug_flag',default=False,type=bool)
    parser.add_argument('--inst', dest='inst',default='acs',type=str)
    parser.add_argument('--filter',dest='filt',default='f814w',type=str)
    parser.add_argument('--filetype',dest='filetype',default='drc',type=str)
    parser.add_argument('--part',dest='part',default=None, type=str)

    args = parser.parse_args()

    main(
        debug_flag = args.debug_flag,
        inst = args.inst,
        filt=args.filt,
        filetype=args.filetype,
        part = args.part
        )
