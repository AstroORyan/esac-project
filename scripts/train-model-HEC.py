'''
Author: David O'Ryan (18.02.2022)
This script is a copy of the local Jupyter Notebook I got working with Zoobot. This script is to be loaded up to the HEC, and just run nice and smoothly.
'''

# Imports
import pandas as pd
import tensorflow as tf

from zoobot.data_utils import image_datasets, create_shards, tfrecord_datasets
from zoobot.estimators import preprocess, define_model
from zoobot.training import training_config, losses
from zoobot import schemas

# Loading in the data.
def main():
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    # check which GPU we're using
    physical_devices = tf.config.list_physical_devices('GPU') 
    print('GPUs: {}'.format(physical_devices))


    # Loading in manifest.
    labelled_catalog = (
        pd.read_csv('/mmfs1/home/users/oryan/Zoobot/data/manifest/zoobot-manifest-hec.csv',index_col=0)
        .rename(columns={'merging_merger':'merger'})
    )

    labelled_catalog_float = (
        labelled_catalog
        .assign(merger_float = labelled_catalog.merger.astype('float64'))
        .assign(id_str_str = labelled_catalog.id_str.astype('str'))
        .drop(columns = ['merger','id_str'])
        .rename(columns={'merger_float' : 'merger', 'id_str_str': 'id_str'})
    )
    # Preparing shard data.
    shard_config = create_shards.Shard_config(
        shard_dir = '/mmfs1/home/users/oryan/Zoobot/tfrecords/',
        size=300
    )

    shard_config.prepare_shards(
        labelled_catalog_float,
        unlabelled_catalog=None,
        test_fraction = 0.20,
        val_fraction=0.10,
        labelled_columns_to_save=['id_str','file_loc','merger']
    )

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

    # Defining our CNN:
    model = define_model.get_model(
        output_dim=2,
        input_size=300,
        crop_size=int(300 * 0.75),
        resize_size=224,
        channels=1
    )

    # Defining our schema:
    schema = schemas.Schema({'merger': ['merger_merger','merger_not_merger']},'merger')

    # Defining our loss function
    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    loss = lambda x, y: multiquestion_loss(x,y)/batch_size

    # Compile our model:
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
    )

    # Configuring our training session:
    train_config = training_config.TrainConfig(
        log_dir = '/mmfs1/home/users/oryan/Zoobot/models/',
        epochs = 50,
        patience=10
    )

    training_config.train_estimator(
        model,
        train_config,
        train_dataset,
        test_dataset,
        val_dataset,
        eager=True
    )

# Initialising
if __name__ == '__main__':
    main()