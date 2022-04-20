'''
This is an updated cutout creator which actually creates images of size 300x300. 

Has the following scalings: ZScaleInvarient(samples=5000, contrast = 0.05) with a LinearStretch()
'''
## Imports
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import glob
import sys

from astroquery.mast import Observations

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize


## Functions
def create_savefolder(folder,id):
    if os.path.exists(f'{folder}/{id}'):
        pass
    else:
        os.mkdir(f'{folder}/{id}')

## Main Function
def main():
    df = pd.read_csv('/mmfs1/home/users/oryan/Zoobot/data/manifest/gz-hubble-local-manifest.csv',index_col=0)
    save_folder = '/mmfs1/scratch/hpc/60/oryan/fromMAST'
    save_dict = {}

    for i in range(len(df)):
        row = df.iloc[i]
        ra = row['RA']
        dec = row['Dec']
        zooniverse_id = row['zooniverse_id']

        if os.path.exists(f'{save_folder}/{zooniverse_id}/{zooniverse_id}.png'):
            save_dict[zooniverse_id] = [f'{save_folder}/thumbnails/{zooniverse_id}_300_300.png']
            continue

        coord = SkyCoord(ra = ra*u.deg, dec = dec*u.deg, frame='fk5')

        obs_table = Observations.query_criteria(
            coordinates = coord,
            radius = 20 * u.arcsec,
            dataproduct_type = 'image',
            obs_collection = 'HLA',
            calib_level = 3,
            type = 'S',
            filters = ['F814W']
        )

        if len(obs_table) < 0.5:
            continue

        parent_obs = pd.DataFrame(
            obs_table
            .to_pandas()
            .obsid
            .value_counts()
            ).reset_index().rename(columns={'index':'parent_obsid'}).drop(columns='obsid')

        data_products = Observations.get_product_list(obs_table)

        download_products = Table.from_pandas(
            data_products
            .to_pandas()
            .query('dataproduct_type == "image"')
            .query('obs_collection == "HLA"')
            .query('type == "C"')
            .query('calib_level == 3')
            .query('productType == "SCIENCE"')
            .merge(parent_obs,on='parent_obsid',how='right')
            .dropna(0,thresh=5)
        )

        create_savefolder(save_folder,zooniverse_id)

        manifest = Observations.download_products(
            download_products,
            download_dir = f'{save_folder}/{zooniverse_id}',
            extension = ['fits']
        )

        for j in manifest:
            file = j['Local Path']
            with fits.open(file,memmap=False) as hdul:
                header = hdul[1].header
                data = hdul[1].data
                hdul.close()
            wcs_out = wcs.WCS(header)

            try:
                cutout = Cutout2D(data,coord,(250,250),wcs=wcs_out,mode='strict')
            except:
                continue

            if len(cutout.data[cutout.data == 0]) > 50:
                continue

            break

        if 'cutout' in locals():
            pass
        else:
            continue

        if len(cutout.data[cutout.data == 0]) > 50:
                continue

        cutout.data[cutout.data == 0] = np.nanmin(data)

        norm = ImageNormalize(
            cutout.data,
            interval=ZScaleInterval(nsamples=7500,contrast = 0.05),
            stretch=LinearStretch(),
            clip=True
        )

        plt.figure(figsize=(12,12))
        plt.imshow(cutout.data,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(f'{save_folder}/{zooniverse_id}/{zooniverse_id}.png', dpi=300, bbox_inches='tight',pad_inches=0)
        plt.close()

        del data, header

        im = Image.open(f'{save_folder}/{zooniverse_id}/{zooniverse_id}.png')
        im_grey = ImageOps.grayscale(im)
        im_grey.thumbnail([300,300])
        im_shape = np.asarray(im_grey).shape
        im_grey.save(f'{save_folder}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png')
        im.close()

        save_dict[zooniverse_id] = [f'{save_folder}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png']

    save_df = pd.DataFrame(save_dict).T.rename(columns={0 : 'thumbnail_path'})
    save_df.to_csv(f'{save_folder}/hubble-thumb-manifest.csv')


## Initialization
if __name__ == '__main__':
    main()