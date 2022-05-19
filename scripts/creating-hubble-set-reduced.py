'''
Author: David O'Ryan
Date: 04/05/2022

This is an updated cutout creator which actually creates images of size 300x300, but uses parrelisation to speed up the process.

Has the following scalings: ZScaleInvarient(samples=5000, contrast = 0.05) with a LinearStretch()
'''
## Imports
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def create_savefolder(folder,id):
    if os.path.exists(f'{folder}/fromMAST/{id}'):
        pass
    else:
        os.mkdir(f'{folder}/fromMAST/{id}')

## Main Function
def main(row):
    ra = row['RA']
    dec = row['Dec']
    zooniverse_id = row['zooniverse_id']

    save_folder = '/mmfs1/scratch/hpc/60/oryan'

    if os.path.exists(f'{save_folder}/full-gz-set/{zooniverse_id}.jpeg'):
        return

    create_savefolder(save_folder,zooniverse_id)

    if ':' in ra:
        coord = SkyCoord(ra, dec, frame='fk5', unit=(u.hourangle, u.deg))
    elif 'm' in ra:
        coord = SkyCoord(ra, dec, frame='fk5',unit = (u.hourangle, u.deg))
    else:
        coord = SkyCoord(ra = float(ra) * u.deg, dec = float(dec) * u.deg, frame='fk5')

    obs_table = Observations.query_criteria(
        coordinates = coord,
        radius = 20 * u.arcsec,
        dataproduct_type = 'image',
        obs_collection = 'HST',
        instrument_name='ACS/WFC',
        calib_level = 3,
        filters = ['F814W']
    )

    if len(obs_table) < 0.5:
        return

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
        .query('obs_collection == "HST"')
        .query('type == "D"')
        .query('calib_level == 3')
        .query('productType == "SCIENCE"')
        .query('productSubGroupDescription == "DRC"')
        .merge(parent_obs,on='parent_obsid',how='right')
        .dropna(0,thresh=5)
    )

    for j in range(len(download_products)):
        manifest = Observations.download_products(
            download_products[j],
            download_dir = f'{save_folder}/fromMAST/{zooniverse_id}',
            extension = ['fits']
        )

        file = list(manifest['Local Path'])[0]
        with fits.open(file,memmap=False) as hdul:
            header = hdul[1].header
            try:
                data = hdul[1].data
            except:
                hdul.close()
                del header
                continue
            wcs_out = wcs.WCS(fobj=hdul, header = header)
            hdul.close()

        try:
            cutout = Cutout2D(data,coord,(150,150),wcs=wcs_out,mode='strict')
        except:
            continue

        if len(cutout.data[cutout.data == 0]) > 50:
            continue

        if len(cutout.data[np.isnan(cutout.data)]) > 0.05*(cutout.data.shape[0]*cutout.data.shape[1]):
            continue
        
        break

    if 'cutout' in locals():
        pass
    else:
        return

    if len(cutout.data[cutout.data == 0]) > 50:
        return

    if len(cutout.data[np.isnan(cutout.data)]) > 0.25*(cutout.data.shape[0]*cutout.data.shape[1]):
        return

    cutout.data[cutout.data == 0] = np.nanmin(data)

    del data

    norm = ImageNormalize(
        cutout.data,
        interval=ZScaleInterval(nsamples=5000,contrast = 0.05),
        stretch=LinearStretch(),
        clip=True
        )

    plt.figure()
    plt.imshow(cutout.data,cmap='Greys_r',norm=norm)
    plt.axis('off')
    figure = plt.gcf()
    figure.set_size_inches(4,4)
    plt.savefig(f'{save_folder}/full-gz-set/{zooniverse_id}.jpeg', dpi=100, bbox_inches='tight',pad_inches=0)
    plt.close()

## Initialization
if __name__ == '__main__':
    df = pd.read_csv('/mmfs1/home/users/oryan/Zoobot/data/manifest/all-gz-validation-set.csv',index_col=0)

    ## Note, can't just input a DataFrame in multiprocessing. Need to create a lazy iterator. Done before setting up Pool:
    for _, i in df.iterrows():
        main(i)