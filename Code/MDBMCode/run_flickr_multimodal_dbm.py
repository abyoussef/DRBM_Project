# Achari Berrada Youssef
# This is an implementation of the Multinomial DBM Architecture.

import numpy as np
import tensorflow as tf
import config
import os

from yadlt.utils import utilities

# Trained from the notebook run_flickr_image_dbm.ipynb
image_layer1_W = np.load("image_dbm_layer_1_W.npy")
image_layer1_b = np.load("image_dbm_layer_1_b.npy")
image_layer2_W = np.load("image_dbm_layer_2_W.npy")
image_layer2_b = np.load("image_dbm_layer_2_b.npy")

# Trained from the notebook run_flickr_text_dbm.ipynb
text_layer1_W = np.load("text_dbm_layer_1_W.npy")
text_layer1_b = np.load("text_dbm_layer_1_b.npy")
text_layer2_W = np.load("text_dbm_layer_2_W.npy")
text_layer2_b = np.load("text_dbm_layer_2_b.npy")



# Input placeholders
img_input = tf.placeholder(tf.float32, [None, 3857], name="image-input")
txt_input = tf.placeholder(tf.float32, [None, 2000], name="text-input")


# ################# #
# Image Forward DBM #
# ################# #

img_W1 = tf.Variable(image_layer1_W)
img_b1 = tf.Variable(image_layer1_b)

img_W2 = tf.Variable(image_layer2_W)
img_b2 = tf.Variable(image_layer2_b)

img_layer1 = tf.add(tf.matmul(img_input, img_W1), img_b1)
img_layer1 = tf.nn.sigmoid(img_layer1)

img_layer2 = tf.add(tf.matmul(img_layer1, img_W2), img_b2)
img_layer2 = tf.nn.sigmoid(img_layer2)



# ################ #
# Text Forward DBM #
# ################ #

txt_W1 = tf.Variable(text_layer1_W)
txt_b1 = tf.Variable(text_layer1_b)

txt_W2 = tf.Variable(text_layer2_W)
txt_b2 = tf.Variable(text_layer2_b)

txt_layer1 = tf.add(tf.matmul(txt_input, txt_W1), txt_b1)
txt_layer1 = tf.nn.sigmoid(txt_layer1)

txt_layer2 = tf.add(tf.matmul(txt_layer1, txt_W2), txt_b2)
txt_layer2 = tf.nn.sigmoid(txt_layer2)


# ############## #
# Multimodal DBM #
# ############## #

joint_representation_units = 512  # number of units in the joint representation layer

multi_W = tf.Variable(tf.truncated_normal(shape=[2048, joint_representation_units], stddev=0.1), name='multimodal-weights')
multi_b = tf.Variable(tf.constant(0.1, shape=[joint_representation_units]), name='multimodal-bias')

multi_input = tf.concat([img_layer2, txt_layer2], 0)
multi_output = tf.nn.sigmoid(tf.add(tf.matmul(multi_input, multi_W), multi_b))

binary_output = utilities.sample_prob(hprobs, np.random.rand(data.shape[0], joint_representation_units))

# ################## #
# Image Backward DBM #
# ################## #

img_layer2b = tf.add(tf.matmul(binary_output, img_W2.T), img_b2)
img_layer2b = tf.nn.sigmoid(img_layer2b)

img_layer1b = tf.add(tf.matmul(img_layer2b, img_W1.T), img_b1)
img_layer1b = tf.nn.sigmoid(img_layer1b)

# ################# #
# Text Backward DBM #
# ################# #

txt_layer2b = tf.add(tf.matmul(binary_output, txt_W2.T), txt_b2)
txt_layer2b = tf.nn.sigmoid(txt_layer2b)

txt_layer1b = tf.add(tf.matmul(txt_layer2b, txt_W1.T), txt_b1)
txt_layer1b = tf.nn.sigmoid(txt_layer1b)

# Now can use img_layer1b (which should be 3587) and txt_layer1b (which should be 2000) to compute the loss
# function. For mean squared error cost:

img_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(img_dataset, img_layer1b))))
txt_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(txt_dataset, txt_layer1b))))

# Optimizer

img_train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(img_cost)
txt_train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(txt_cost)

# ####################### #
# Code to train the model #
# ####################### #

# train_set and train_ref should be loaded, maybe from the labeled data directory
labeled1 = np.load(os.path.join(config.flickr_labeled_path, "combined-00001-of-00100.npy"))
labeled2 = np.load(os.path.join(config.flickr_labeled_path, "combined-00002-of-00100.npy"))
labeled3 = np.load(os.path.join(config.flickr_labeled_path, "combined-00003_0-of-00100.npy"))
labeled = np.concatenate((labeled1, labeled2, labeled3), axis=0)

num_epochs = 5
batch_size = 64

shuff = zip(train_set, train_ref)

with tf.Session() as sess:
    for i in range(num_epochs):
        np.random.shuffle(shuff)
        batches = [_ for _ in utilities.gen_batches(shuff, batch_size)]

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            # run image train
            sess.run(img_train_step, feed_dict={img_input: x_batch, image_ref: y_batch})
            # run text train
            sess.run(txt_train_step, feed_dict={txt_input: x_batch, text_ref: y_batch})

        if validation_set is not None:
            img_loss = sess.run(img_cost, feed_dict={img_input: img_validation_input, img_ref: img_validation_ref})
            txt_loss = sess.run(txt_cost, feed_dict={txt_input: txt_validation_input, txt_ref: txt_validation_ref})
            print("Image loss: %f" % (img_loss))
            print("Text loss: %f" % (txt_loss))


