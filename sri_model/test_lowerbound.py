"""
Test script for continual anomaly detection.
"""

import datasets
import models
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constants
pic_size = 28*28
enc_neurons = [400, 300, 200, 100]
code_size = 50
dec_neurons = [100, 200, 300, 400]
batch_size = 128
learning_rate = 0.001
epochs = 25
eval_posterior_samples = 100
gen_sample_batch_size = 512
thd = -170.0
normal_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
anomaly = [0]

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
posterior_samples_ph = tf.placeholder(tf.int32, shape=[])
gen_sample_batch_size_ph = tf.placeholder(tf.int32, shape=[])
thd_ph = tf.placeholder(tf.float32, shape=[])
VAE = models.dense_VAE(batch_data, batch_labels, pic_size, enc_neurons, code_size, dec_neurons, posterior_samples_ph, gen_sample_batch_size_ph, thd_ph)

# Build optimizer
learning_rate_ph = tf.placeholder(tf.float32)
opt = tf.train.AdamOptimizer(learning_rate_ph)
update = opt.minimize(-1.0*tf.reduce_mean(VAE.elbo))

# Start tf session
sess = tf.Session()
saver = tf.train.Saver()

for run in range(10):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    metrics_list = []

    for i in range(9):
        # Load data for training
        data = datasets.split_anomaly_landsat([normal_classes[i]], anomaly)
        [train_data, train_labels] = data.get_train_samples()
        train_data = train_data / 255.0

        # Reinitialize optimizer
        sess.run([tf.variables_initializer(opt.variables()), tf.local_variables_initializer()])

        # Train VAE
        utils.train(locals(), 0, saver, VAE, train_data, train_labels, epochs, learning_rate, batch_size)

        # Load data for evaluation
        data = datasets.split_anomaly_mnist(normal_classes[0:i+1], anomaly)
        [eval_data, eval_labels] = data.get_eval_samples()
        eval_data = eval_data / 255.0

        # Detect anomalies
        metrics = utils.detect_anomalies(locals(), VAE, eval_data, eval_labels, thd, eval_posterior_samples, show_metrics=True, plot_elbo_hist=False)
        metrics_list.append(metrics)

    # Load data for evaluation
    data = datasets.split_anomaly_landsat(normal_classes[0:i+1], anomaly)
    [eval_data, eval_labels] = data.get_eval_samples()
    eval_data = eval_data / 255.0

    # Detect anomalies on first task
    metrics = utils.detect_anomalies(locals(), VAE, eval_data, eval_labels, thd, eval_posterior_samples, show_metrics=True, plot_elbo_hist=False)
    metrics_list.append(metrics)

    # Save metrics
    hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
            "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
            "eval_posterior_samples": eval_posterior_samples, "gen_sample_batch_size": gen_sample_batch_size, "thd": thd,\
            "normal_classes": normal_classes, "anomaly": anomaly}
    utils.metrics_saver(metrics_list, VAE.metric_names, hp_dict, "./log/Lower_bound")
