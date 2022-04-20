'''
This file will take in the full list of the coordinates of every gzm subject (primary and secondary) and create cutouts just from those images.
'''

## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

from astroquery.mast import Observations

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize

## Functions

## Main Function
def main():
    df = pd.read_csv('/mmfs1/home/users/oryan/Zoobot/data/manifest/gz-mergers-coords-all-seperated.csv')
    save_path = '/mmfs1/scratch/hpc/60/oryan/fromMAST'
    save_dict = {}

    for i in range(len(df)):
        row = df.iloc[i]

        ra = row.RA
        dec = row.Dec

        zooniverse_id = row.Names.strip()

        coord = SkyCoord(ra = ra * u.deg, dec = dec * u.deg, frame='icrs')

        obs_table = Observations.query_criteria(
            coordinates = coord,
            radius = 20 * u.arcsec,
            dataproduct_type='image',
            obs_collection = 'HLA',
            calib_level = 3,
            type='S',
            filters = ['F814W', 'F606W', 'F435W', 'g', 'r', 'i']
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
            .merge(parent_obs, on='parent_obsid',how='right')
            .dropna(0,thresh=5)
        )

        if os.path.exists(f'{save_path}/{zooniverse_id}'):
            pass
        else:
            os.mkdir(f'{save_path}/{zooniverse_id}')

        manifest = Observations.download_products(
            download_products,
            download_dir = f'{save_path}/{zooniverse_id}',
            extension = ['fits']
        )

        for j in manifest:
            file = j['Local Path']
            try:
                with fits.open(file) as hdul:
                    header = hdul[1].header
                    data = hdul[1].data
                    hdul.close()
                wcs_out = wcs.WCS(header)
            except:
                continue

            try:
                cutout = Cutout2D(data,coord,(1000,1000),wcs=wcs_out,mode='strict')
                break
            except:
                continue

            if len(cutout.data[cutout.data == 0]) > 50:
                continue


            del data, header
        
        if 'cutout' in locals():
            pass
        else:
            continue

        if len(cutout.data[cutout.data == 0]) > 50:
                continue

        cutout.data[cutout.data == 0] = np.nanmin(cutout.data[cutout.data > 0])

        norm = ImageNormalize(cutout.data,interval=ZScaleInterval(nsamples=7500,contrast=0.05),stretch=LinearStretch(),clip=False)

        plt.figure(figsize=(12,12))
        plt.imshow(cutout.data,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(f'{save_path}/{zooniverse_id}/{zooniverse_id}.png',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close()

        im = Image.open(f'{save_path}/{zooniverse_id}/{zooniverse_id}.png')
        im_grey = ImageOps.grayscale(im)
        im_grey.thumbnail([300,300])

        im_shape = np.asarray(im_grey).shape
        im_grey.save(f'{save_path}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png')
        im.close()

        save_dict[zooniverse_id] = [f'{save_path}/thumbnails/{zooniverse_id}_{im_shape[0]}_{im_shape[1]}.png']

    save_df = pd.DataFrame(save_dict).T.rename(columns={0 : 'thumbnail_loc'})
    save_df.to_csv(f'{save_path}/gzm-image-manifest.csv')



## Initialization
if __name__ == '__main__':
    main()