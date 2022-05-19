'''
This algorithm will take in a bunch of fits files and find the coordinates associated with, returning them as well as the cutouts.
'''
import logging

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize

import numpy as np
import random
import os

from astroquery.mast import Catalogs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_func(data, wcs, coords):
    sample_list = list(coords)
    counter = 0
    bad_count = 0
    while counter < 50:
        sample_coord = coords[random.choice(sample_list)]
        try:
            cutout = Cutout2D(data,sample_coord,(150,150),wcs=wcs,mode='strict')
        except:
            continue
        if np.isnan(np.max(cutout.data)):
            continue
        elif np.max(cutout.data) > 5:
            counter += 1
            bad_count += 1
        elif np.max(cutout.data) <= 5:
            counter += 1
    
    if bad_count >= 25:
        return True
    else:
        return False

def get_coords(file):
    data, wcs = get_fits(file)
    central_coord = wcs.pixel_to_world(int(data.shape[0]/2), int(data.shape[1]/2))

    attempt = 0
    downloaded = False
    while attempt < 3 and downloaded == False:
        try:
            catalogue_data = Catalogs.query_region(central_coord, radius=0.05, catalog='HSC')
            downloaded = True
        except: 
            attempt += 1

    if attempt == 3:
        catalogue_data = []

    if len(catalogue_data) == 0:
        logging.info(f'No sources found in {file}. Investigate?')
        return None, None, None
    ras = catalogue_data['MatchRA']
    decs = catalogue_data['MatchDec']
    matchids = catalogue_data['MatchID']

    coords = SkyCoord(ra = ras * u.deg, dec = decs * u.deg, frame = 'icrs')

    coords_dict = dict(zip(matchids, coords))
    
    return data, wcs, coords_dict

def get_fits(filepath):
    with fits.open(filepath, mmep=False) as hdul:
        wcs = WCS(hdul[1].header)
        data = hdul[1].data
        hdul.close()
    return data, wcs


def cutout_creation(coord, matchid, data, wcs, save_folder, file):
    if os.path.exists(f'{save_folder}/{file}/{matchid}.jpeg'):
        return

    try:
        cutout = Cutout2D(data, coord,(150,150),wcs=wcs,mode='strict')
    except:
        return

    tmp = len(cutout.data[np.isnan(cutout.data)])
    if tmp > 10:
        return

    norm = ImageNormalize(
        cutout.data,
        interval = ZScaleInterval(nsamples=5000,contrast=0.05),
        stretch=LinearStretch(),
        clip=True
    )

    fig = plt.figure()
    plt.imshow(cutout.data,cmap='Greys_r',norm=norm)
    plt.axis('off')
    figure = plt.gcf()
    figure.set_size_inches(4,4)
    plt.savefig(f'{save_folder}/{file}/{matchid}.jpeg',dpi=100,bbox_inches='tight',pad_inches=0)
    plt.clf()
    plt.close()