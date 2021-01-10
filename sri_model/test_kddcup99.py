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
pic_size = 41
enc_neurons = [400, 300, 200, 100]
code_size = 50
dec_neurons = [100, 200, 300, 400]
batch_size = 1024*16
learning_rate = 0.001
epochs = 50
eval_posterior_samples = 100
gen_sample_batch_size = 1024*16
thd = -160.0
normal_classes = [11, 1, 2, 4, 5, 7, 8, 10, 12, 13, 15, 16, 17, 19, 21, 22]
anomaly = [0, 6, 9, 14, 18, 20]
log_path = "./log/vae"

# Create dataset and batching using tensorflow
data_ph = tf.placeholder(tf.float32, name="data_ph")
labels_ph = tf.placeholder(tf.float32, name="labels_ph")
batch_size_ph = tf.placeholder(tf.int64, name="batch_size_ph")
shufflebuffer_ph = tf.placeholder(tf.int64, name="shufflebuffer_ph")
data_ds = tf.data.Dataset.from_tensor_slices(data_ph)
labels_ds = tf.data.Dataset.from_tensor_slices(labels_ph)
dataset = tf.data.Dataset.zip((data_ds, labels_ds)).shuffle(shufflebuffer_ph).batch(batch_size_ph)
iterator = dataset.make_initializable_iterator()
[batch_data, batch_labels] = iterator.get_next()

# Create VAE
posterior_samples_ph = tf.placeholder(tf.int32, shape=[], name="posterior_samples_ph")
gen_sample_batch_size_ph = tf.placeholder(tf.int32, shape=[], name="gen_sample_batch_size_ph")
# thd_ph = tf.placeholder(tf.float32, shape=[], name="thd_ph")
thd_ph=0.0
b_replay_ph = tf.placeholder(tf.bool, shape=[], name="b_replay_ph")

VAE = models.dense_VAE(batch_data, batch_labels, b_replay_ph, pic_size,  enc_neurons, code_size, dec_neurons, gen_sample_batch_size_ph, thd_ph, learning_rate)

# Build optimizer
learning_rate_ph = tf.placeholder(tf.float32)
opt = tf.train.AdamOptimizer(learning_rate_ph)
update = opt.minimize(-1.0*tf.reduce_mean(VAE.elbo))

# Start tf session
sess = tf.Session()
saver = tf.train.Saver()


# Create tensorboard
hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
        "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
        "normal_classes": normal_classes, "anomaly": anomaly, "log_path": log_path}
[fw, log_path_dt] = utils.create_tensorboard(sess, log_path, hp_dict)


for run in range(10):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    metrics_list = []

    # Load data for training
    data = datasets.split_anomaly_kddcup99(normal_classes, anomaly)
    [train_data, train_labels] = data.get_train_samples()
    [eval_data, eval_labels] = data.get_eval_samples()

    for i in range(16):
        # Reinitialize optimizer
        sess.run([tf.variables_initializer(opt.variables()), tf.local_variables_initializer()])

        # Train VAE
        # utils.train(locals(), i, saver, VAE, train_data, train_labels, epochs, learning_rate, batch_size)
        utils.train(sess, i,VAE, epochs, iterator, train_data, train_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
        
        # Detect anomalies
        utils.initialize_eval_data(sess, iterator, eval_data, eval_labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph, thd_ph)
        # metrics = utils.detect_anomalies(locals(), VAE, eval_data, eval_labels, thd, eval_posterior_samples, show_metrics=True, plot_elbo_hist=False)
        metrics = utils.detect_anomalies(sess, VAE, fw, i, thd_ph)
        metrics_list.append(metrics)

    # Save metrics
    hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
            "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
            "eval_posterior_samples": eval_posterior_samples, "gen_sample_batch_size": gen_sample_batch_size, "thd": thd,\
            "normal_classes": normal_classes, "anomaly": anomaly}
    utils.metrics_saver(metrics_list, VAE.metric_names, hp_dict, "./log/Degeneration")
