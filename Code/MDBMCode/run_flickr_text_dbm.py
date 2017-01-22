# Achari Berrada Youssef

# This script is for training the Text specific Deep Boltzmann Machine. 
import os
import numpy as np
import tensorflow as tf
import config

import dbm

# Load flickr text
dataset = np.load("flickr_text_dataset.npy")
D = np.sum(dataset, axis=1)  # length of each document

model = dbm.DBM(
    main_dir="flickr_rbm", do_pretrain=True, layers=[1024, 1024],
    models_dir=config.models_dir, data_dir=config.data_dir, summary_dir=config.summary_dir,
    learning_rate=[0.001, 0.001], momentum=0.9, num_epochs=[1, 1], batch_size=[64, 64],
    stddev=0.1, verbose=1, gibbs_k=[1, 1], model_name="flickr_dbm",
    finetune_learning_rate=0.01, finetune_enc_act_func=[tf.nn.sigmoid, tf.nn.sigmoid],
    finetune_dec_act_func=[tf.nn.sigmoid, tf.nn.sigmoid], finetune_num_epochs=1, finetune_batch_size=128,
    finetune_opt='momentum', finetune_loss_func="mean_squared", finetune_dropout=0.5, noise=["rsm", "bin"],
)

# Initialize documents lengths
for i, _ in enumerate(model.rbms):
    model.rbms[i].D = D

# Pretraining phase 
model.pretrain(dataset[:500], dataset[500:1000])

# Fit and save the txt-DBM parameters 
# I put save_dbm_text_params as a quick hack to save the parameters of this dbm as a numpy array
model.fit(dataset[:500], dataset[500:1000], dataset[1500:2000], dataset[2000:2500], save_dbm_text_params=True)


# Output : 
    # text_dbm_layer_1_W.npy
    # text_dbm_layer_1_b.npy
    # text_dbm_layer_2_W.npy
    # text_dbm_layer_2_b.npy