# Achari Berrada Youssef

# This is the script for training the Image Specific Deep Boltzmann Machine
import os
import numpy as np
import tensorflow as tf
import config

import dbm

# Load all the .npy files in the flickr unlabeled images directory into flickr_u

flickr_u = np.array([])
files_count = 0

for f in os.listdir(config.flickr_unlabeled_path):
    if files_count == 10:  # only ten files for now
        break
    files_count += 1
    if f[-4:] == ".npy":
        t = np.load(os.path.join(config.flickr_unlabeled_path, f))
        if flickr_u.shape == (0,):
            flickr_u = t
        else:
            flickr_u = np.concatenate((flickr_u, t), axis=0)

model = dbm.DBM(
    main_dir="flickr_rbm", do_pretrain=True, layers=[1024, 1024],
    models_dir=config.models_dir, data_dir=config.data_dir, summary_dir=config.summary_dir,
    learning_rate=[0.001, 0.001], momentum=0.9, num_epochs=[1, 1], batch_size=[64, 64],
    stddev=0.1, verbose=1, gibbs_k=[1, 1], model_name="flickr_dbm",
    finetune_learning_rate=0.01, finetune_enc_act_func=[tf.nn.sigmoid, tf.nn.sigmoid],
    finetune_dec_act_func=[tf.nn.sigmoid, tf.nn.sigmoid], finetune_num_epochs=5, finetune_batch_size=128,
    finetune_opt='momentum', finetune_loss_func="mean_squared", finetune_dropout=0.5, noise=["gauss", "bin"],
)

trainX = flickr_u[:3000]
testX = flickr_u[3000:3500]


model.pretrain(trainX, testX)

model.fit(trainX, testX, trainX[:10], testX[:10], save_dbm_image_params=True) 

# Output : 
    # image_dbm_layer_1_W.npy
    # image_dbm_layer_1_b.npy
    # image_dbm_layer_2_W.npy
    # image_dbm_layer_2_b.npy

