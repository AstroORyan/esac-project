'''
Author: David O'Ryan
Date: 13.04.2022
The big difference here is that this script will call MAST and get the images directly with wget.
'''

## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import glob
import sys
import time

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D, block_reduce
from astropy.table import Table, join
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize, AsinhStretch, LogStretch, ManualInterval
from astropy.stats import sigma_clip
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel,Tophat2DKernel, convolve

## Functions

## Main Function
def main():
    df = pd.read_csv('/mmfs1/home/users/oryan/Zoobot/data/manifest/gz-hubble-local-manifest.csv', index_col = 0)[:5]
    for i in range(len(df)):
        row = df.iloc[i]
        ra = row['RA']
        dec = row['Dec']
        zooniverse_id = row['zooniverse_id']

        coord = SkyCoord(ra = ra * u.deg, dec = dec * u.deg, frame = 'fk5')

        obs_table = Observations.query_criteria(
            coordinates = coord,
            radius = 2 * u.arcmin,
            dataproduct_type = 'image',
            obs_collection = 'HLA',
            calib_level = 3,
            type='S',
            filters = ['F814W']
        )

        cutoff = np.max(obs_table['t_obs_release']) - 500

        wanted_obs = Table.from_pandas(
            obs_table
            .to_pandas()
            .sort_values('t_obs_release',ascending=False)
            .query('t_obs_release >= @cutoff')
        )

        data_products = Observations.get_product_list(wanted_obs)

        download_products = Table.from_pandas(
            data_products
            .to_pandas()
            .query('dataproduct_type == "image"')
            .query('obs_collection == "HLA"')
            .query('type == "C"')
            .query('calib_level == 3')
            .query('productType == "SCIENCE"')
        )

        n_download = len(download_products)

        if n_download > 10:
            print(f'Download number greater than tolerance. It is {n_download}')
            sys.exit()
        
        if os.path.exists(f'/mmfs1/scratch/hpc/60/oryan/fromMAST/{zooniverse_id}/'):
            pass
        else:
            os.mkdir(f'/mmfs1/scratch/hpc/60/oryan/fromMAST/{zooniverse_id}/')
            time.sleep(5)

        manifest = Observations.download_products(
            download_products,
            download_dir = f'/mmfs1/scratch/hpc/60/oryan/fromMAST/{zooniverse_id}/'
        )
        
        print('FITS files downloaed.')

## Initialization
if __name__ == '__main__':
    main()