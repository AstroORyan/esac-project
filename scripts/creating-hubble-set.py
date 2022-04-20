'''
Author: David O'Ryan
Date: 05/04/2022

This script will create the Hubble training data from the previously created manifest. This will:
    
    1. Query the HLA/HST.
    2. Find the obervation we're looking for.
    3. Download the relevent fits file.
    4. Use that to make the cutout.
    5. Delete the first file and the directory containing it.
    6. Repeat until gone through manifest.
'''
## Imports
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import subprocess
import sys
from pathlib import Path
import time
from PIL import Image
import numpy as np
import shutil
import glob

from astroquery.mast import Observations

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy.table import Table, join
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize

## Functions

## Main Function
def main():
    
    manifest_path = '/home/users/oryan/Zoobot/data/manifest/gz-hubble-local-manifest.csv'
    save_path = '/mmfs1/storage/users/oryan/hubble-cutouts'

    manifest_df = pd.read_csv(manifest_path)

    paths_dict = {}
    mosaicked_dict = {}

    for i in range(0,len(manifest_df)):
        row = manifest_df.iloc[i]
        ra = row['RA']
        dec = row['Dec']
        zooniverse_id = row['zooniverse_id']
        tel_fil = 'F814W'
        
        if os.path.exists(f'/mmfs1/storage/users/oryan/hubble-cutouts/{zooniverse_id}.png'):
            paths_dict[str(zooniverse_id)] = [f'/mmfs1/storage/users/oryan/hubble-cutouts/{zooniverse_id}.png']
            print(f'File {zooniverse_id} exists. Skipping...')
            continue

        coord = SkyCoord(ra = ra*u.deg, dec = dec * u.deg, frame = 'fk5')

        connected = False
        counter = 0
        while counter <= 10 and connected == False:
            try:
                obs_table = Observations.query_region(coord,radius=0.002 * u.deg)
                connected = True
            except:
                counter += 1
    

        if len(obs_table.to_pandas().query('filters == "F814W" and instrument_name == "ACS/WFC"')) < 0.5:
            print(f'WARNING: Also no F814W observations for {zooniverse_id}. Skipping...')
            continue

        data_product = Table.from_pandas(
            obs_table
            .to_pandas()
            .sort_values('t_exptime')
            .query('filters == @tel_fil')
            .query('calib_level == 3')
            .query('dataproduct_type == "image"')
            .query('instrument_name == "ACS/WFC"')
        )

        obs_id = data_product['obs_id']

        no_obs = True
        counter = 0

        if len(obs_id) < 0.5:
            print(f'WARNING: No valid observations for {zooniverse_id} using given query. Skipping...')
            continue

        while no_obs and counter < len(obs_id):
            single_obs = Observations.query_criteria(obs_collection='HST',obs_id=obs_id)
            if len(single_obs) > 0.5:
                no_obs = False
            else:
                counter += 1

        if no_obs:
            print(f'WARNING: Found no obsverations for {zooniverse_id}. Skipping...')
            continue

        if counter == len(obs_id) - 1:
            print(f'WARNING: Found now observations for {zooniverse_id}. Skipping...')
            continue

        data_products = Observations.get_product_list(single_obs)
        if len(data_products) < 0.5:
            print(f'WARNING: No data products! Check code...')
            sys.exit()

        data_products_download = Observations.filter_products(
            data_products,
            productType = ['SCIENCE'],
            extension = ['fits']
        )

        products_download = join(data_product, data_products_download, keys='obs_id')

        products_download.rename_columns(['obs_collection_1'],['obs_collection'])

        download_manifest = Observations.download_products(
            products_download,
            productType='SCIENCE',
            obs_collection = ['HST'],
            download_dir = '/mmfs1/scratch/hpc/60/oryan/hubble-processed-data',
            extension = ['fits']
        )

        fits_path = list(download_manifest['Local Path'])
        skip_flag = False

        if len(fits_path[0]) < 1.5:
            with fits.open(fits_path) as hdul:
                counts = hdul[1].data
                head = hdul[1].header
                hdul.close()
            w = wcs.WCS(head)
            cutout = Cutout2D(counts, coord, (300,300), wcs=w, mode='partial').data
        else:
            for i in range(len(fits_path)):
                with fits.open(fits_path[i]) as hdul:
                    counts = hdul[1].data
                    head = hdul[1].header
                    hdul.close()
                w = wcs.WCS(head)
                cutout = Cutout2D(counts, coord, (300,300), wcs=w, mode='partial').data
                if not np.isnan(cutout.data).any():
                    break
                elif i == len(fits_path) - 1:
                    print(f'WARNING: Will need to mosaic for image {zooniverse_id}')
                    skip_flag = True
                    mosaicked_dict[zooniverse_id] = fits_path[0]

        if skip_flag:
            continue

        vmax = np.max(cutout)
        mean = np.mean(cutout/vmax)
        
        std_dev = 0
        for i in range(cutout.shape[0]):
            for j in range(cutout.shape[1]):
                std_dev += ((cutout[i,j]/vmax) - mean)**2
        RMS_Con = np.sqrt(std_dev / (cutout.shape[0]*cutout.shape[1]))

        norm = ImageNormalize(cutout, interval=ZScaleInterval(nsamples=5000,contrast=RMS_Con),stretch=AsinhStretch(),clip=True)

        filepath = os.path.join(save_path, f'{zooniverse_id}.png')
        plt.figure(figsize=(10,10))
        plt.imshow(cutout,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(filepath,bbox_inches = 'tight', pad_inches=0,dpi=300)
        plt.close()

        paths_dict[str(zooniverse_id)] = [filepath]

        if len(glob.glob('/mmfs1/scratch/hpc/60/oryan/hubble-processed-data/mastDownload/HST/*')) > 10:
            print('Deleting Fits files...')
            time.sleep(10)
            shutil.rmtree('/mmfs1/scratch/hpc/60/oryan/hubble-processed-data/mastDownload/HST/')

    save_manifest = pd.DataFrame(paths_dict).T.rename(columns={0 : 'grey_scale_image'})
    mosaicked_manifest = pd.DataFrame(mosaicked_dict).T.rename(columns={0:'fits_path'})

    output_manifest = manifest_df.merge(save_manifest,left_on='zooniverse_id',right_index=True,how='left')

    output_manifest.to_csv(os.path.join(save_path,'gz-hubble-combined-manifest.csv'))
    mosaicked_manifest.to_csv(os.path.join(save_path, 'gz-hubble-requiring_mosaicks.csv'))

    print('Algorithm Complete.')         

## Initialization
if __name__ == '__main__':
    main()