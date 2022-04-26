'''
This is a second take at trying to do finetuning for Zoobot. 

Need to load in the Zoobot trained model and then prepare it in the same way as was done by GZ.
'''
## Imports
import os
import logging
import glob
import random
import shutil

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot.shared import schemas, label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import preprocess, define_model, alexnet_baseline, small_cnn_baseline
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset
from zoobot.tensorflow.training import training_config
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.estimators import custom_layers

## Functions

## Main Function
def main():
    requested_img_size = 300
    batch_size = 3
    file_format = 'png'
    
    folder = '/mmfs1/home/users/oryan/Zoobot/data/manifest'

    df = (pd.read_csv(f'{folder}/hubble-thumb-manifest-checked.csv',index_col = 0)
    .reset_index()
    )
    
    paths = list(df['thumbnail_path'])
    labels = list(df['interacting'].astype(int))
    logging.info('Labels: \n{}'.format(pd.value_counts(labels)))

    paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.2, random_state=42)
    assert set(paths_train).intersection(set(paths_val)) == set()

    raw_train_dataset = image_datasets.get_image_dataset(paths_train,file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size,labels=labels_val)

    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],
        input_size=requested_img_size,
        normalise_from_uint8=True,
        input_channels=1,
        make_greyscale=True,
        permute_channels=False
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset,preprocess_config)

    folder = '/mmfs1/home/users/oryan/Zoobot/pretrained_models'
    pretrained_checkpoint = f'{folder}/replicated_train_only_greyscale_tf/replicated_train_only_greyscale_tf/checkpoint'

    crop_size = int(requested_img_size * 0.75)
    resize_size = 224

    log_dir = '/mmfs1/home/users/oryan/Zoobot/models/logs'

    logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
    base_model = define_model.load_model(
        pretrained_checkpoint,
        include_top=False,
        input_size = requested_img_size,
        crop_size = crop_size,
        resize_size = resize_size,
        output_dim=None,
        expect_partial=True
    )

    base_model.trainable = False

    new_head = tf.keras.Sequential([
        layers.InputLayer(input_shape=(7,7,1280)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.75),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(1,activation='sigmoid',name='sigmoid_output')
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(requested_img_size, requested_img_size,1)),
        base_model,
        new_head
    ])

    epochs = 80
    loss = tf.keras.losses.binary_crossentropy

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    model.summary()

    train_config = training_config.TrainConfig(
        log_dir=log_dir,
        epochs=epochs,
        patience=int(epochs/6)
    )
    
    training_config.train_estimator(
        model,
        train_config,
        train_dataset,
        val_dataset
    )

    losses = []
    for _ in range(5):
        losses.append(model.evaluate(val_dataset)[0])
    logging.info('Mean validation loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))

    logging.info('Finetuning Complete.')

## Initialization
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    main()