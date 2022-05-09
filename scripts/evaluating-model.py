'''
Author (more like editor): David O'Ryan
Date: 05/05/2022

This code is taken directly from the finetune_advanced script by Mike Walmsley. I have removed everything in it to do with fine tuning though, and will just use it to evaluate
how the model is doing from a validation set.
'''
## Imports
from ctypes import resize
import os
import logging
import time
import json
import pandas as pd

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from zoobot.tensorflow.estimators import preprocess, define_model
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.data_utils import image_datasets

## Functions
def evaluate_performance(
    model,
    test_dataset,
    run_name,
    log_dir,
    batch_size,
    train_dataset_size
):
    losses = []
    accuracies = []
    for _ in range(5):
        test_metrics = model.evaluate(test_dataset.repeat(3), verbose=0)
        losses.append(test_metrics[0])
        accuracies.append(test_metrics[1])
    logging.info(f'Mean test loss {np.mean(losses)} (var {np.var(losses)})')
    logging.info(f'Mean test accuracy {np.mean(accuracies)} (var {np.var(accuracies)})')

    predictions = model.predict(test_dataset).astype(float).squeeze()
    logging.info(predictions)
    labels = np.concatenate([label_batch.numpy().astype(float) for _, label_batch in test_dataset]).squeeze()
    logging.info(labels)
    results = {
        'batch_size'
        'mean_loss'
        'mean_acc'
        'predictions' : predictions.tolist(),
        'labels': labels.tolist(),
        'train_dataset_size' : int(train_dataset_size),
        'log_dir' : log_dir,
        'run_name' : run_name
    }
    json_name = f'{run_name}_result_timestamped_{train_dataset_size}_{np.random.randint(10000)}.json'
    json_loc = os.path.join(log_dir, json_name)

    with open(json_loc,'w') as f:
        json.dump(results, f)
    
    logging.info(f'Results save to {json_loc}')

## Main Function
def main():
    ### Prepping the Data
    folder = '/mmfs1/home/users/oryan/Zoobot/data/manifest'
    file_format = 'png'
    requested_img_size = 300
    batch_size = 64
    run_name = 'trained_model_valuation'

    df = (pd.read_csv(f'{folder}/large-training-set-labelled.csv',index_col = 0)
    .reset_index()
    )

    paths = list(df['thumbnail_path'])
    labels = list(df['interaction'].astype(int))
    logging.info('Labels: \n{}'.format(pd.value_counts(labels)))

    _, paths_val, _, labels_val = train_test_split(paths, labels, test_size = 0.2, random_state = 42)

    dataset_size = len(paths_val)

    raw_val_dataset = image_datasets.get_image_dataset(
        paths_val, 
        file_format = file_format, 
        requested_img_size = requested_img_size,
        batch_size=batch_size,
        labels=labels_val
        )

    preprocess_config = preprocess.PreprocessingConfig(
        label_cols = ['label'],
        input_size = requested_img_size,
        normalise_from_uint8 = True,
        input_channels = 3,
        make_greyscale=True,
        permute_channels = False
    )

    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)

    ### Building the Model
    folder = '/mmfs1/home/users/oryan/Zoobot/pretrained_models'
    pretrained_checkpoint = f'{folder}/replicated_train_only_greyscale_tf/replicated_train_only_greyscale_tf/checkpoint'
    finetuned_checkpoint = '/mmfs1/home/users/oryan/Zoobot/finetuned-model/full/checkpoint'

    crop_size = int(requested_img_size * 0.75)
    resize_size = 224

    log_dir = '/mmfs1/home/users/oryan/Zoobot/validation/'

    logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
    #### These lines must be IDENTICAL to how the model was trained. Otherwise I'm worried it'd work, and outputs would suck.
    base_model = define_model.load_model(
        pretrained_checkpoint,
        include_top = False,
        input_size = requested_img_size,
        crop_size = crop_size,
        resize_size = resize_size,
        output_dim = None,
        channels = 1,
        expect_partial = True
    )

    para_relu = layers.PReLU()

    new_head = tf.keras.Sequential([
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
        tf.keras.layers.InputLayer(input_shape = (requested_img_size, requested_img_size, 1)),
        base_model,
        new_head
    ])
    model.load_weights(finetuned_checkpoint).expect_partial()
    
    loss = tf.keras.losses.binary_crossentropy
    model.compile(
        loss = loss,
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    evaluate_performance(
        model=model,
        test_dataset=val_dataset,
        run_name=run_name + '_valuation',
        log_dir = log_dir,
        batch_size = batch_size,
        train_dataset_size = dataset_size
    )

## Initialization
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(levelname)s@ %(message)s')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    main()