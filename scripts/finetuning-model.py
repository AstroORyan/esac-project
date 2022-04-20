'''
Example script to fine tune our model which has been trained on the SDSS data.
'''
## Imports
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import logging
import time
import json

from zoobot.data_utils import tfrecord_datasets
from zoobot.estimators import preprocess, define_model
from zoobot.training import training_config
from zoobot.transfer_learning import utils

## Functions
def evaluate_performance(model, test_dataset, run_name, log_dir, batch_size, train_dataset_size):
    losses = []
    accuracies = []
    for _ in range(5):
        test_metrics = model.evaluate(test_dataset.repeat(3),verbose = 0)
        losses.append(test_metrics[0])
        accuracies.append(test_metrics[1])
    logging.info('Mean test loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    logging.info('Mean test accuracy: {:.3f} (var {:.4f})'.format(np.mean(accuracies), np.var(accuracies)))

    predictions = model.test_dataset.astype(float).squeeze()
    logging.info(predictions)
    labels = np.concatenate([label_batch.numpy().astype(float) for _, label_batch in test_dataset]).squeeze()
    logging.info(labels)
    results = {
        'batch_size' : int(batch_size),
        'mean_loss': float(np.mean(losses)),
        'mean_acc': float(np.mean(accuracies)),
        'predictions':predictions.tolist(),
        'labels': labels.tolist(),
        'train_dataset_size': int(train_dataset_size),
        'log_dir': log_dir,
        'run_name': str(os.path.basename(log_dir))
    }

    json_name = '{}_result_timestamped_{}_{}.json'.format(run_name, train_dataset_size, np.random.randint(100000))
    json_loc = os.path.join(log_dir,json_name)

    with open(json_loc,'w') as f:
        json.dump(results, f)

    logging.info(f'Results saved to {json_loc}')

## Main Function
def main():
    # Reset up training data so that we can train the new head.
    train_records = '/mmfs1/home/users/oryan/Zoobot/tfrecords/train_shards/s300_shard_0.tfrecord'
    test_records = '/mmfs1/home/users/oryan/Zoobot/tfrecords/test_shards/s300_shard_0.tfrecord'
    val_records = '/mmfs1/home/users/oryan/Zoobot/tfrecords/val_shards/s300_shard_0.tfrecord'

    columns_to_save = ['merger']
    batch_size = 64
    raw_train_dataset = tfrecord_datasets.get_tfrecord_dataset(train_records,columns_to_save,batch_size,shuffle=True)
    raw_test_dataset = tfrecord_datasets.get_tfrecord_dataset(test_records,columns_to_save,batch_size,shuffle=False)
    raw_val_dataset = tfrecord_datasets.get_tfrecord_dataset(val_records,columns_to_save,batch_size,shuffle=False)

    # Preprocessing data.
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['merger'],
        input_size=300,
        normalise_from_uint8=True,
        make_greyscale=True,
        permute_channels=False
    )

    train_dataset = preprocess.preprocess_dataset(raw_train_dataset,preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset,preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset,preprocess_config)

    train_dataset_size = len(train_dataset)

    pretrained_checkpoint = '/mmfs1/home/users/oryan/Zoobot/models/checkpoint'
    
    # Hard code these, I'm pretty sure they must be identical to what I did in the training script.
    requested_img_size = 300
    crop_size = int(300 * 0.75)
    resize_size = 224
    channels = 1

    # Loading my pretrained model.
    base_model = define_model.get_model(
        pretrained_checkpoint,
        include_top=False,
        input_size = requested_img_size,
        crop_size=crop_size,
        resize_size=resize_size,
        output_dim=None,
        channels=channels,
        expect_partial=True
    )
    base_model.trainable = False

    # Define the new head following the finetune_advanced.py example.
    new_head = tf.keras.Sequential([
        layers.InputLayer(input_shape = (7,7,1280)),
        layers.GlobalAceragePooling2D(),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(1,activation='sigmoid',name='sigmoid_output')
    ])

    # Attach head on the pretrained model.
    model = tf.keras.Sequentual([
        tf.keras.layers.InputLayer(input_shape=(requested_img_size, requested_img_size, channels)),
        base_model,
        new_head
    ])

    # Now, we retrain the model only training the new head:
    epochs = 50
    patience = 10

    logging.info('Epochs: {}'.format(epochs))
    logging.info('Patience: {}'.format(patience))

    model.compile(
        loss = tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    model.summary(print_fn=logging.info)

    run_name = 'example_run_{}'.format(time.time())
    log_dir = os.parth.join('/mmfs1/home/users/oryan/Zoobot/models/test_run',run_name)
    log_dir_head = os.path.join(log_dir,'head_only')
    for d in [log_dir,log_dir_head]:
        if not os.path.isdir(d):
            os.mkdir(d)


    train_config = training_config.TrainConfig(
        log_dir=log_dir_head,
        epochs=epochs,
        patience=patience
    )

    training_config.train_estimator(
        model,
        train_config,
        train_dataset,
        val_dataset
    )

    evaluate_performance(
        model=model,
        test_dataset=test_dataset,
        run_name=run_name + '_transfer',
        log_dir=log_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size
    )

    # Now that we have trained the head, we need to train the model as a whole. 
    logging.info('Unfreezing layers')
    utils.unfreeze_model(model,unfreeze_names=['top'])

    logging.info('Recompiling with lower learning rate and trainable upper layers')

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )

    model.summary(print_fn=logging.info)

    log_dir_full = os.path.join(log_dir,'full')
    train_config_full = training_config.TrainConfig(
        log_dir=log_dir_full,
        epochs=epochs,
        patience=patience 
    )

    training_config.train_estimator(
        model,
        train_config_full,
        train_dataset,
        val_dataset
    )

    logging.info('Finetuning complete')

    evaluate_performance(
        model=model,
        test_dataset = test_dataset,
        run_name = run_name + '_finetuned',
        log_dir=log_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size
    )
    

## Initialisation
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    
    main()