'''
Author: David O'Ryan
Date: 04/05/2022

This script will be to use in DataLabs and with the Hubble Source Catalogue. So, we will load in a fits file from the archive and find its WCS. From this WCS, we will extract
the region. This region will then be used with the HSC to check what's in it. These coordinates will be saved, and appended into a dictionary. Will decide on an output size
and stop when we have that many objects in the manifest. Save this file, and wipe memory. Then, create new manifest in this way.

Should split up all objects into .csvs of 500,000 objects. If I get more than, say, 100 .csvs, stop process altogether and rething what we're doing. 

It will need to call my filename creator script if it goes down the route of running through the archive.
'''
## Imports
from astroquery.mast import Catalogs

## Functions

## Main Function
def main():
    catalog_data = Catalogs.query_object()

## Initialisation
if __name__ == '__main__':
    main()