'''
author: oryan

This script will use the new SciServer package that I've just installed to download the cutouts from SkySerer from a list of RA and DECs. Let's go!!
'''

# Imports
import typer
from pathlib import Path
import pandas as pd
from SciServer import SkyServer
from astropy import units as u
from astropy.coordinates import SkyCoord
from PIL import Image
from tqdm import tqdm

# Functions

# Main Function
def main(
    manifest: Path = typer.Option(..., '-M',help='Please specify the path to the manifest of objects to get cutouts of. Requires columns of RA and DEC.'),
    save_folder: Path = typer.Option(...,'-S',help='Please specify the save folder for the cutouts you are downloading.'),
    dataRelease: str = typer.Option('DR14', '-D',help='Please specify the SDSS release you want to get the cutouts from.'),
    opt: str = typer.Option('','-O',help='Please specify any options you want for the cutout (see https://rdrr.io/github/sciserver/SciScript-R/man/SkyServer.getJpegImgCutout.html) for all options. Default: blank.'),
    query: str = typer.Option('','-Q',help = 'Please specify any extra SQL commands you want. Default: blank.')
    ):

    df = pd.read_csv(manifest,index_col=0)

    output_dict = {'id_str':[],'filepath':[]}

    for i in tqdm(range(len(df))):
        ra = df.ra.iloc[i]
        dec = df.dec.iloc[i]
        counter = 0
        
        while counter < 10:
            try:
                img_arr = SkyServer.getJpegImgCutout(ra=ra, dec=dec, width=300,height=300,scale=0.4,dataRelease=dataRelease,opt=opt,query=query)
            except:
                print(f'Struggling to get cutout. Retrying... Beginning attempt {counter+1}')
                counter += 1

        if counter >= 10:
            print(f'Failed to download {df.dr7objid.iloc[i]}. Continuing...')
            continue
    
        img = Image.fromarray(img_arr)

        #img.thumbnail((224,224))

        filename = f'{str(df.dr7objid.iloc[i])}.png'
        save_path = Path.joinpath(save_folder,filename)

        img.save(save_path,quality=95)

        output_dict['id_str'].append(df.dr7objid.iloc[i])
        output_dict['filepath'].append(save_path)

        del ra, dec, img
    
    output_df = pd.DataFrame(output_dict)

    export_df = df.merge(output_df,left_on='dr7objid',right_on='id_str',how='left')

    manifest_save = Path.joinpath(save_folder,'testing-manifest-loc.csv')
    export_df.to_csv(manifest_save)



# Initialisation
if __name__ == '__main__':
    typer.run(main)