'''
This script will simply load in the Zoobot model.
'''
import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys

import tensorflow as tf
from tensorflow.keras import layers

from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.predictions import predict_on_dataset

def check_image_res(df):
    file_paths = df.file_loc
    for filepath in file_paths:
        im = Image.open(filepath)
        im_arr = np.asarray(im)
        if im_arr.shape[-1] == 4:
            im_grey = im.convert('RGB')
            im_grey.save(filepath)
            im_grey.close()
        im.close()

def get_manifest(path,file):
    cutout_paths = glob.glob(f'{path}/{file}/*.jpeg')

    cutouts_dict = {
        os.path.basename(image_path).replace('.jpeg','') : [image_path]
        for image_path in cutout_paths
    }

    cutouts_df = (
        pd.DataFrame(cutouts_dict)
        .T
        .reset_index()
        .rename(columns={'index':'id_str',0:'file_loc'})
    )

    return cutouts_df

def make_predictions(ds, model, j, save_folder):
    cut = 0.55

    n_samples = 1
    label_cols = ['interacting']
    save_loc = f'{save_folder}/predictions/pred_{j}.csv'

    predictions = predict_on_dataset.predict(ds, model, n_samples, label_cols, save_loc, ret_flag = True)

    predictions_export = (
        predictions
        .assign(matchid = predictions.id_str.apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
        .assign(binary_prediction = predictions.interacting_pred.apply(lambda x: 1 if ast.literal_eval(x)[0] > cut else 0))
        .rename(columns={'id_str' : 'file_path'})
    )

    return predictions_export

def process_images(ds, initial_size):
    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols = [],
        input_size = initial_size,
        make_greyscale = True,
        normalise_from_uint8 = True
    )

    processed_ds = preprocess.preprocess_dataset(ds, preprocessing_config)

    return processed_ds

def get_predictions(df, model, save_folder, i):
    file_format = 'jpeg'
    initial_size = 300
    batch_size = 8

    raw_image_ds = image_datasets.get_image_dataset(
        [x for x in df.file_loc], file_format, initial_size, batch_size
    )

    processed_image_ds = process_images(raw_image_ds, initial_size)

    predictions = make_predictions(processed_image_ds, model, i, save_folder)

    return predictions


def load_model():
    initial_size = 300
    crop_size = int(0.75 * initial_size)
    resize_size = 224

    pretrained_checkpoint = '/media/home/my_workspace/test-exp-archive/pretrained-model/checkpoint' 
    finetuned_checkpoint = '/media/home/my_workspace/test-exp-archive/finetuned-model/checkpoint'

    base_model = define_model.load_model(
        pretrained_checkpoint,
        include_top = False,
        input_size = initial_size,
        crop_size = crop_size,
        resize_size = resize_size,
        output_dim = None,
        channels = 1,
        expect_partial = True
    )

    para_relu = layers.PReLU()

    model_head = tf.keras.Sequential([
        layers.InputLayer(input_shape=(7,7,1280)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64,activation=para_relu),
        #layers.Dropout(0.75),
        layers.Dense(64,activation='elu'),
        layers.Dropout(0.25),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(64,activation = para_relu),
        layers.Dropout(0.25),
        layers.Dense(1,activation='sigmoid',name='sigmoid_output')
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(initial_size, initial_size, 1)),
        base_model,
        model_head
    ])

    model.load_weights(finetuned_checkpoint).expect_partial()

    return model