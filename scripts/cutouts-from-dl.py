'''
Script to generate cutouts directly from data labs. Have training set of 16,000 I want to use. 

Let's goooooo.
'''
## Imports
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from get_paths import creating_obs_filepath


## Functions

## Main Function
def main():
    manifest = pd.read_csv('/media/home/my_workspace/manifests/all-gz-training-set.csv')
    save_folder = '/media/home/my_workspace/cutouts/'

    for i in tqdm(range(len(manifest))):
        row = manifest.iloc[i]
        ra = row.RA 
        dec = row.Dec
        zooniverse_id = row.zooniverse_id
        req_filter = 'f814w'

        if os.path.exists(f'{save_folder}/reduced_cutouts/{zooniverse_id}.png'):
            continue

        if ':' in ra:
            coord = SkyCoord(ra, dec, frame='fk5', unit=(u.hourangle, u.deg))
        elif 'm' in ra:
            coord = SkyCoord(ra, dec, frame='fk5',unit = (u.hourangle, u.deg))
        else:
            coord = SkyCoord(ra = float(ra) * u.deg, dec = float(dec) * u.deg, frame='fk5')

        filepath = creating_obs_filepath.create_url(coord, req_filter, collection = ['HST','HLA'])

        if filepath is None:
            continue

        with fits.open(filepath,mmap=False) as hdul:
            header = hdul[1].header
            data = hdul[1].data
            hdul.close()
        
        w = WCS(header)

        cutout = Cutout2D(data, coord, (100,100), wcs=w, mode='strict')

        del data

        if len(cutout.data[cutout.data == 0]) > 50:
            continue

        if len(cutout.data[np.isnan(cutout.data)]) > 0.25*(cutout.data.shape[0]*cutout.data.shape[1]):
            continue

        norm = ImageNormalize(
            cutout.data,
            interval=ZScaleInterval(nsamples=5000,contrast = 0.05),
            stretch=LinearStretch(),
            clip=True
        )

        plt.figure(figsize=(12,12))
        plt.imshow(cutout.data,cmap='Greys_r',norm=norm)
        plt.axis('off')
        plt.savefig(f'{save_folder}/tmp_pngs/{zooniverse_id}.png', dpi=300, bbox_inches='tight',pad_inches=0)
        plt.close()

        del header, cutout

        im = Image.open(f'{save_folder}/tmp_pngs/{zooniverse_id}.png')
        im_grey = im.convert('RGB')
        im_grey.thumbnail([300,300])
        im_shape = np.asarray(im_grey).shape
        im_grey.save(f'{save_folder}/reduced_cutouts/{zooniverse_id}.png')
        im.close()



## Initialization
if __name__ == '__main__':
    main()