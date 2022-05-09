'''
Author: David O'Ryan
Date: 27.04.2022

Want this to be a very quick test at just using the basic Zoobot DeCALs Model to find interacting galaxies.

Will be accompanied by a Notebook of some kind. 
'''
## Imports
import tensorflow as tf
import logging
import os

from zoobot.tensorflow.estimators import preprocess, define_model
from zoobot.tensorflow.predictions import predict_on_dataset
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.shared import label_metadata

## Functions

## Main Functions
def main():
    file_format = 'png'

    image_paths = predict_on_dataset.paths_in_folder('/mmfs1/storage/users/oryan/hubble-cutouts/', file_format = file_format, recursive = False)

    assert len(image_paths) > 0
    assert os.path.isfile(image_paths[0])

    initial_size = 300
    batch_size = 64
    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in image_paths], file_format, initial_size, batch_size)

    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols = [],
        input_size = initial_size,
        make_greyscale=True,
        normalise_from_uint8 = True
    )

    image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)

    crop_size = int(initial_size * 0.75)
    resize_size = 224
    channels = 3

    checkpoint_loc = '/mmfs1/home/users/oryan/Zoobot/pretrained_models/replicated_train_only_greyscale_tf/replicated_train_only_greyscale_tf/checkpoint'

    model = define_model.load_model(
        checkpoint_loc = checkpoint_loc,
        include_top = True,
        input_size = initial_size,
        crop_size = crop_size,
        resize_size = resize_size,
        expect_partial = True
    )

    label_cols = label_metadata.decals_label_cols

    save_loc = '/mmfs1/home/users/oryan/Zoobot/predictions/decals-model/validation-predictions.csv'
    n_samples = 5
    predict_on_dataset.predict(image_ds, model, n_samples, label_cols, save_loc)

## Initialization
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    main()