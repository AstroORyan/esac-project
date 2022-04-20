## Imports
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageOps
import time
import shutil
import glob

from astroquery.mast import Observations

from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy import wcs
from astropy.nddata import Cutout2D, block_reduce
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
from astropy.io import fits

import astropy.units as u

## Functions

## Main Function
def main():
    manifest_loc = '/mmfs1/home/users/oryan/Zoobot/data/manifest/gz-mergers-coords.csv'
    save_path_files = '/mmfs1/storage/users/oryan/mergers-cutout/'
    save_path_thumbnails = '/mmfs1/storage/users/oryan/mergers-thumbnails/'

    manifest_df = pd.read_csv(manifest_loc, index_col=0)

    save_dic = {}
    missing_dic = {}

    for i in range(len(manifest_df)):
        row = manifest_df.iloc[i]
        prim_RA = row['Prim_RA'] 
        sec_RA = row['Sec_RA']
        prim_Dec = row['Prim_DEC']
        sec_Dec = row['Sec_DEC']
        name = row['Names'].strip()

        prim_coords = SkyCoord(ra=prim_RA * u.deg, dec = prim_Dec * u.deg, frame = 'fk5')
        sec_coords = SkyCoord(ra=sec_RA * u.deg, dec = sec_Dec * u.deg, frame = 'fk5')
        coords = [prim_coords, sec_coords]

        filters = []

        #for j in range(2):
        obs_table = Observations.query_region(coords[0], radius=0.002 * u.deg)
        if obs_table.to_pandas().filters.value_counts()['g'] > 0:
            filters.append('g')
        elif obs_table.to_pandas().filters.value_counts()['r'] > 0:
            filters.append('r')
        elif obs_table.to_pandas().filters.value_counts()['z'] > 0:
            filters.append('z')
        else:
            print(f'Need to look for other filters for galaxy {name}')
            missing_dic[name] = 'Missing'
            continue
        

        data_product = Table.from_pandas(
            obs_table
            .to_pandas()
            .sort_values('t_exptime')
            .query('filters == @filters[0]')
            .query('intentType == "science"')
            .query('dataproduct_type == "image"')
        )

        obs_ids = (data_product[0]['obs_id'])

        observations = Observations.query_criteria(obs_id=obs_ids,filters=filters)

        data_products = Observations.get_product_list(observations)

        data_products_download = Observations.filter_products(
            data_products,
            productType=['SCIENCE'],
            dataRights = ['PUBLIC'],
            extension='fits'
        )

        products_download = join(data_product, data_products_download, keys='obs_id')

        products_download.rename_columns(['obs_collection_1'],['obs_collection'])

        manifest = Observations.download_products(
            data_products_download,
            productType = 'SCIENCE',
            download_dir = '/mmfs1/scratch/hpc/60/oryan/mergers-processed-data/',
            #obs_collection = ['HLA'],
            extension=['fits']

        )

        fits_files = manifest['Local Path'][0]


        with fits.open(fits_files,memmap=False) as hdul:
            counts = hdul[1].data
            w = wcs.WCS(hdul[1].header)

        cutout = Cutout2D(counts, coords[0],(1500,1500),wcs=w,mode='trim').data
        red_im = block_reduce(cutout, 5)

        norm = ImageNormalize(red_im, interval=ZScaleInterval(nsamples=5000, contrast=0.02),stretch=LinearStretch(),clip=True)

        filepath = os.path.join(save_path_files, f'{name}.png')
        plt.figure()
        plt.imshow(red_im,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(filepath,bbox_inches = 'tight', pad_inches=0,dpi=300)
        plt.close()

        im = Image.open(filepath)
        im_gray = ImageOps.grayscale(im)
        im_gray.thumbnail([300,300])
        thumb_path = os.path.join(save_path_thumbnails, f'{name}_thumbnail.png')
        im_gray.save(thumb_path)

        save_dic[name] = thumb_path

        if len(glob.glob('/mmfs1/scratch/hpc/60/oryan/mergers-processed-data/mastDownload/PS1/*')) > 10:
            print('Deleting Fits files...')
            time.sleep(10)
            shutil.rmtree('/mmfs1/scratch/hpc/60/oryan/mergers-processed-data/mastDownload/PS1/*')

        del counts, w
        
    save_df = pd.DataFrame(save_dic).T.rename(columns='grey_scale_image')
    miss_df = pd.DataFrame(missing_dic).T.rename(columns='grey_scale_image')
    export_df = pd.concat([save_df, miss_df])
    export_df.to_csv(os.path.join(save_path_thumbnails,'gz-merger-saved.csv'))

## Initialization
if __name__ == '__main__':
    main()