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
from PIL import Image, ImageOps
import numpy as np
import glob

from astroquery.mast import Observations

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy.table import Table, join
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize

## Functions
def zero_check(im):
    if len(im[im == 0]) > 50:
        return True
    else:
        return False

def centre_check(im):
    centre = im[int(im.shape[0]/2) - 3: int(im.shape[0]/2) + 3, int(im.shape[1]/2) - 3:int(im.shape[1]/2) + 3]
    if np.mean(centre) > np.mean(im):
        return False
    else:
        return True

## Main Function
def main():
    
    manifest_path = 'C:/Users/oryan/Documents/esac-project/manifests/gz-hubble-local-manifest.csv'
    save_path = 'C:/Users/oryan/Documents/zoobot_new/preprocessed-cutouts/fromMAST'

    manifest_df = pd.read_csv(manifest_path)

    paths_dict = {}

    for i in range(0,len(manifest_df)):
        row = manifest_df.iloc[i]
        ra = row['RA']
        dec = row['Dec']
        zooniverse_id = row['zooniverse_id']
        tel_fil = 'F814W'
        
        if os.path.exists(f'C:/Users/oryan/Documents/zoobot_new/preprocessed-cutouts/fromMAST/{zooniverse_id}.png'):
            paths_dict[str(zooniverse_id)] = [f'C:/Users/oryan/Documents/zoobot_new/preprocessed-cutouts/fromMAST/{zooniverse_id}.png']
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
            single_obs = Observations.query_criteria(obs_collection=['HLA','HST'],obs_id=obs_id)
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

        for i in range(len(products_download)):
            download_manifest = Observations.download_products(
                products_download[i],
                productType='SCIENCE',
                obs_collection = ['HST','HLA'],
                download_dir = f'C:/Users/oryan/Documents/zoobot_new/preprocessed-cutouts/fromMAST/{zooniverse_id}',
                extension = ['fits']
            )
    
            fits_path = list(download_manifest['Local Path'])
            skip_flag = False
    
            with fits.open(fits_path[0]) as hdul:
                counts = hdul[1].data
                head = hdul[1].header
                hdul.close()
            w = wcs.WCS(head)
            cutout = Cutout2D(counts, coord, (100,100), wcs=w, mode='partial').data
            
            centre_flag = centre_check(cutout)
            zero_flag = zero_check(cutout)
            
            if zero_flag:
                continue
            if centre_flag:
                continue
            
            break

        norm = ImageNormalize(cutout, interval=ZScaleInterval(nsamples=7500,contrast=0.05),stretch=LinearStretch(),clip=True)

        filepath = os.path.join(save_path, f'{zooniverse_id}.png')
        plt.figure(figsize=(10,10))
        plt.imshow(cutout,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(filepath,bbox_inches = 'tight', pad_inches=0,dpi=300)
        plt.close()

        im = Image.open(f'{save_path}/{zooniverse_id}/{zooniverse_id}.png')
        im_grey = ImageOps.grayscale(im)
        im_grey.thumbnail([300,300])
        im_shape = np.asarray(im_grey).shape
        im_grey.save(f'{save_path}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png')
        im.close()

        paths_dict[zooniverse_id] = [f'{save_path}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png']


    save_manifest = pd.DataFrame(paths_dict).T.rename(columns={0 : 'grey_scale_image'})

    output_manifest = manifest_df.merge(save_manifest,left_on='zooniverse_id',right_index=True,how='left')

    output_manifest.to_csv(os.path.join(save_path,'gz-hubble-combined-manifest.csv'))

    print('Algorithm Complete.')         

## Initialization
if __name__ == '__main__':
    main()