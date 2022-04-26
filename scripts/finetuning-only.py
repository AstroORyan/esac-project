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
import json

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
def evaluate_performance(model, test_dataset, run_name, log_dir, batch_size, train_dataset_size):
    losses = []
    accuracies = []
    for _ in range(5):
        test_metrics = model.evaluate(test_dataset.repear(3),verbose = 0)
        losses.append(test_metrics[0])
        accuracies.append(test_metrics[1])
    logging.info(f'Mean test loss {np.mean(losses)} (var {np.var(losses)})')
    logging.info(f'Mean test accuracy {np.mean(accuracies)} (var {np.var(accuracies)})')

    predictions = model.predict(test_dataset).astype(float).squeeze()
    logging.info(predictions)
    labels = np.concatenate([label_batch.numpy().astype(float) for _, label_batch in test_dataset]).squeeze()
    logging.info(labels)
    results = {
        'batch_size': int(batch_size),
        'mean_loss': float(np.mean(losses)),
        'mean_acc': float(np.mean(accuracies)),
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
    run_name = '2022-04-26-HEC'

    requested_img_size = 300
    batch_size = 64
    file_format = 'png'
    
    folder = '/mmfs1/home/users/oryan/Zoobot/data/manifest'

    df = (pd.read_csv(f'{folder}/hubble-thumb-manifest-checked.csv',index_col = 0)
    .reset_index()
    )
    df_test = (
        pd.read_csv(f'{folder}/hubble-thumb-valid-checked.csv',index_col=0)
        .reset_index()
    )
    
    paths = list(df['thumbnail_path'])
    labels = list(df['interacting'].astype(int))
    logging.info('Labels: \n{}'.format(pd.value_counts(labels)))

    paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.2, random_state=42)
    paths_test = list(df_test['thumbnail_path'])
    labels_test = list(df_test['interacting'])
    assert set(paths_train).intersection(set(paths_val)) == set()
    assert set(paths_train).intersection(set(paths_test)) == set()
    assert set(paths_val).intersection(set(paths_test)) == set()

    raw_train_dataset = image_datasets.get_image_dataset(paths_train,file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size,labels=labels_val)
    raw_test_dataset = image_datasets.get_image_dataset(paths_test, file_format=file_format, requested_img_size = requested_img_size, batch_size = batch_size, labels=labels_test)

    train_dataset_size = len(paths_train)

    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],
        input_size=requested_img_size,
        normalise_from_uint8=True,
        input_channels=3,
        make_greyscale=True,
        permute_channels=False
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset,preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

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
        #layers.Dropout(0.75),
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

    epochs = 250
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
        patience=int(epochs/10)
    )
    
    training_config.train_estimator(
        model,
        train_config,
        train_dataset,
        val_dataset,
        verbose=1
    )

    losses = []
    for _ in range(5):
        losses.append(model.evaluate(val_dataset)[0])
    logging.info('Mean validation loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))

    evaluate_performance(
        model=model,
        test_dataset=test_dataset,
        run_name=run_name + '_transfer_1',
        log_dir=log_dir,
        batch_size = batch_size,
        train_dataset_size = train_dataset_size
    )

    logging.info('Unfreezing Layers')
    utils.unfreeze_model(model, unfreeze_names = ['top'])

    logging.info('Recompiling with lower learning rate and trainable upper layers')
    model.compile(
        loss = loss,
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
        metrics=['accuracy']
    )

    model.summary(print_fn = logging.info)

    log_dir_full = os.path.join(log_dir, 'full')
    train_config_full = training_config.TrainConfig(
        log_dir=log_dir_full,
        epochs=epochs,
        patience=30
    )

    training_config.train_estimator(
        model,
        train_config_full,
        train_dataset,
        val_dataset
    )

    logging.info('Finetuning Complete.')

    evaluate_performance(
        model=model,
        test_dataset=test_dataset,
        run_name=run_name + '_finetuned',
        log_dir=log_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size
    )

## Initialization
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s: %(message)s')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    main()