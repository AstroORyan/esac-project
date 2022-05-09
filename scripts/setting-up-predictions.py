'''
Script to run that will begin to make predictions upon galaxy zoo data.
'''
## Imports
import os
import logging
import glob
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, regularizers


from zoobot.shared import label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_dataset

## Functions

## Main Function
def main():
    # First, gotta get the data!
    manifest_path = '/mmfs1/home/users/oryan/Zoobot/data/manifest/validation-set-manifest-checked.csv'
    manifest = pd.read_csv(manifest_path,index_col=0)

    file_format = 'png'

    image_paths =  manifest['filepath']

    ## Check these are valid...
    assert len(image_paths) > 0
    assert os.path.isfile(image_paths[0])

    ### Must now load as a tf.dataset
    initial_size = 300
    batch_size = 64
    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in image_paths], file_format, initial_size, batch_size)

    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],
        input_size=initial_size,
        input_channels=3,
        make_greyscale=True,
        normalise_from_uint8 = True,
        permute_channels = False
    )
    image_ds = preprocess.preprocess_dataset(raw_image_ds,preprocessing_config)

    ## Now, we load in our model that has been finetuned.
    crop_size=int(initial_size * 0.75)
    resize_size = 224
    channels=1

    pretrained_checkpoint = '/mmfs1/home/users/oryan/Zoobot/pretrained_models/replicated_train_only_greyscale_tf/replicated_train_only_greyscale_tf/checkpoint'
    finetuned_loc = '/mmfs1/home/users/oryan/Zoobot/finetuned-model/full/checkpoint'

    base_model = define_model.load_model(
        checkpoint_loc = pretrained_checkpoint,
        include_top = False,
        input_size = initial_size,
        crop_size = crop_size,
        resize_size = resize_size,
        expect_partial=True
    )

    para_relu = layers.PReLU()

    new_head = tf.keras.Sequential([
        layers.InputLayer(input_shape=(7,7,1280)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64,activation=para_relu),
        #layers.Dropout(0.75),
        layers.Dense(64,activation='elu'),
        layers.Dropout(0.75),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(64,activation = para_relu),
        layers.Dropout(0.75),
        layers.Dense(1,activation='sigmoid',name='sigmoid_output')
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(initial_size,initial_size,channels)),
        base_model,
        new_head
    ])
    model.load_weights(finetuned_loc).expect_partial()

    label_cols = ['interacting']

    # model = define_model.load_model(
    #    checkpoint_loc=finetuned_loc,
    #    include_top=True,
    #    input_size=initial_size,
    #    crop_size=crop_size,
    #    resize_size=resize_size,
    #    expect_partial=True
    # )

    ## Now, we make predictions!
    n_samples=1
    save_loc = '/mmfs1/home/users/oryan/Zoobot/predictions/validation-test.csv'

    predict_on_dataset.predict(image_ds,model,n_samples,label_cols,save_loc)

    logging.info(f'Save the predictions to {save_loc}')



## Initialization
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    main()