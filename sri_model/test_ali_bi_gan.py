"""
Test script for continual anomaly detection.
"""

import datasets
import models
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Constants
pic_size = 28*28
code_size = 50
enc_neurons = [400, 300, 200, 100]
dec_neurons = [100, 200, 300, 400]
disc_neurons = [400, 300]
batch_size = 128
learning_rate = 0.001
epochs = 30
thd = 100
normal_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
anomaly = [0]
log_path = "./log/ali_bi_gan"

# Create dataset and batching using tensorflow
data_ph = tf.placeholder(tf.float32)
labels_ph = tf.placeholder(tf.float32)
batch_size_ph = tf.placeholder(tf.int64)
shufflebuffer_ph = tf.placeholder(tf.int64)
data_ds = tf.data.Dataset.from_tensor_slices(data_ph)
labels_ds = tf.data.Dataset.from_tensor_slices(labels_ph)
dataset = tf.data.Dataset.zip((data_ds, labels_ds)).shuffle(shufflebuffer_ph).batch(batch_size_ph)
iterator = dataset.make_initializable_iterator()
[batch_data, batch_labels] = iterator.get_next()

# Create VAE
gen_sample_batch_size_ph = tf.placeholder(tf.int32, shape=[])
b_replay_ph = tf.placeholder(tf.bool, shape=[])
model = models.dense_ali_bi_gan(batch_data, batch_labels, pic_size, code_size, enc_neurons, dec_neurons, disc_neurons, gen_sample_batch_size_ph, learning_rate, thd, b_replay_ph)

# Start tf session
sess = tf.Session(config=config)

# Initialize variables
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# Create tensorboard
hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
        "disc_neurons": disc_neurons, "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
        "thd": thd, "normal_classes": normal_classes, "anomaly": anomaly, "log_path": log_path}
[fw, log_path_dt] = utils.create_tensorboard(sess, log_path, hp_dict)

for i in range(9):
    sess.run(tf.variables_initializer(model.disc_fake.vars+model.disc_real.vars))
    # Load data for training
    data = datasets.split_anomaly_mnist([normal_classes[i]], anomaly)
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0

    # Train VAE
    utils.train(sess, i, model, epochs, iterator, train_data, train_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)

    # Load data for evaluation
    data = datasets.split_anomaly_mnist(normal_classes[0:i+1], anomaly)
    [eval_data, eval_labels] = data.get_eval_samples()
    eval_data = eval_data / 255.0

    # Detect anomalies
    utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
    metrics = utils.detect_anomalies(sess, model, fw, i)
    utils.plot_gen_imgs(sess, model, 64, log_path_dt, i)
    utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
    utils.plot_recon_imgs(sess, model, 64, log_path_dt, i)
    utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
    utils.plot_anomscore(sess, model, log_path_dt, i)
