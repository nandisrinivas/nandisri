"""
Test script for continual anomaly detection.
"""

import datasets
import models
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Constants
pic_size = 28*28
enc_neurons = [400, 300, 200, 100]
code_size = 50
dec_neurons = [100, 200, 300, 400]
batch_size = 128
learning_rate = 0.001
epochs = 25
thd = -150.0
normal_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
anomaly = [0]
log_path = "./log/vae_serial"

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
model = models.dense_VAE(batch_data, batch_labels, b_replay_ph, pic_size, enc_neurons, code_size, dec_neurons, gen_sample_batch_size_ph, thd, learning_rate)

# Start tf session
sess = tf.Session(config=config)

# Initialize variables
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# Create tensorboard
hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
        "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
        "thd": thd, "normal_classes": normal_classes, "anomaly": anomaly, "log_path": log_path}
[fw, log_path_dt] = utils.create_tensorboard(sess, log_path, hp_dict)

for run in range(10):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    metrics_list = []

    # Load data for training
    data = datasets.split_anomaly_mnist(normal_classes, anomaly)
    [train_data, train_labels] = data.get_train_samples()
    [eval_data, eval_labels] = data.get_eval_samples()
    train_data = train_data / 255.0
    eval_data = eval_data / 255.0

    for i in range(9):
        # Train VAE
        utils.train(sess, i, model, epochs, iterator, train_data, train_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)

        # Detect anomalies
        utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
        metrics = utils.detect_anomalies(sess, model, fw, i)
        utils.plot_gen_imgs(sess, model, 64, log_path_dt, i)
        utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
        utils.plot_recon_imgs(sess, model, 64, log_path_dt, i)
        utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
        utils.plot_anomscore(sess, model, log_path_dt, i)
