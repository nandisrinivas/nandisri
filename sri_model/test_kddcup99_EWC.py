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
epochs = 25
eval_posterior_samples = 100
gen_sample_batch_size = 1024*16
thd = -10.0
lmbd = 100.0
normal_classes = [11, 1, 2, 4, 5, 7, 8, 10, 12, 13, 15, 16, 17, 19, 21, 22]
anomaly = [0, 6, 9, 14, 18, 20]

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
    var_list = VAE.enc.vars + VAE.dec.vars
    prev_elas = []
    for v in var_list:
        prev_elas.append(np.zeros(v.shape, dtype=np.float32))

    for i in range(16):
        # Load data for training
        data = datasets.split_anomaly_kddcup99([normal_classes[i]], anomaly)
        [train_data, train_labels] = data.get_train_samples()

        # Reinitialize optimizer
        sess.run([tf.variables_initializer(opt.variables()), tf.local_variables_initializer()])

        # Train VAE
        utils.train(locals(), 0, saver, VAE, train_data, train_labels, epochs, learning_rate, batch_size)
        [new_elas, means] = utils.get_EWC_reg(locals(), saver, VAE, train_data, 10000)
        [elasticity, prev_elas] = utils.combine_elasticity(new_elas, prev_elas)
        update = utils.create_EWC_update(VAE, elasticity, means, lmbd, opt)

        # Load data for evaluation
        data = datasets.split_anomaly_kddcup99(normal_classes[0:i+1], anomaly)
        [eval_data, eval_labels] = data.get_eval_samples()
        # Detect anomalies
        metrics = utils.detect_anomalies(locals(), VAE, eval_data, eval_labels, thd, eval_posterior_samples, show_metrics=True, plot_elbo_hist=False)
        metrics_list.append(metrics)

    # Save metrics
    hp_dict = {"pic_size": pic_size, "enc_neurons": enc_neurons, "code_size": code_size, "dec_neurons": dec_neurons,\
            "batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs,\
            "eval_posterior_samples": eval_posterior_samples, "gen_sample_batch_size": gen_sample_batch_size, "thd": thd,\
            "normal_classes": normal_classes, "anomaly": anomaly}
    utils.metrics_saver(metrics_list, VAE.metric_names, hp_dict, "./log/EWC")
