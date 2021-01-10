"""
These classes provide datasets for training and testing of neural networks
in the context of continual learning.
"""

import abc
import numpy as np
import tensorflow as tf

class dataset:
    """
    Base class for dataset.
    """
    __metaclass__ = abc.ABCMeta

    # Constructor
    def __init__(self):
        print("Constructor of dataset class...")

    def get_train_samples(self, idx = None):
        # Return training samples based on a index
        if (idx is None):
            return self.train
        else:
            return (self.train[0][idx], self.train[1][idx])

    def get_eval_samples(self, idx = None):
        # Return training samples based on a index
        if (idx is None):
            return self.eval
        else:
            return (self.eval[0][idx], self.eval[1][idx])

    def filter(self, x, l):
        tmp = np.zeros(x.shape, dtype=np.bool)
        for i in range(x.shape[0]):
            if (x[i] in l):
                tmp[i] = True
        return tmp

    @abc.abstractmethod
    def get_dataset_name(self):
        pass

class mnist(dataset):
    """
    Class for the mnist dataset.
    """

    def __init__(self):
        # Load MNIST data
        [self.train, self.eval] = tf.keras.datasets.mnist.load_data()

    def get_dataset_name(self):
        return "MNIST"

class split_anomaly_mnist(mnist):
    """
    Class for the mnist dataset with task to detect anomolies.
    """

    def __init__(self, normal = [0], anomaly = [1]):
        # Load MNIST data
        [train, eva] = tf.keras.datasets.mnist.load_data()
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], normal)], np.zeros_like(train[1][self.filter(train[1], normal)]))
        train_b = (train[0][self.filter(train[1], anomaly)], np.ones_like(train[1][self.filter(train[1], anomaly)]))
        eval_a = (eva[0][self.filter(eva[1], normal)], np.zeros_like(eva[1][self.filter(eva[1], normal)]))
        eval_b = (eva[0][self.filter(eva[1], anomaly)], np.ones_like(eva[1][self.filter(eva[1], anomaly)]))
        # Stack both classes ontop of each other
        self.train = train_a
        self.eval = (np.concatenate((eval_a[0], eval_b[0], train_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1], train_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split anomaly MNIST"

class fashion_mnist(mnist):
    """
    Class for the fashion mnist dataset.
    """
    def __init__(self):
        # Load fashion mnist data
        [self.train, self.eval] = tf.keras.datasets.fashion_mnist.load_data()

    def get_dataset_name(self):
        return "Fashion MNIST"

class split_anomaly_fashion_mnist(fashion_mnist):
    """
    Class for the fashion mnist dataset with task to classify two classes.
    """

    def __init__(self, normal = [0], anomaly = [1]):
        # Load fashion mnist data
        [train, eva] = tf.keras.datasets.fashion_mnist.load_data()
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], normal)], np.zeros_like(train[1][self.filter(train[1], normal)]))
        train_b = (train[0][self.filter(train[1], anomaly)], np.ones_like(train[1][self.filter(train[1], anomaly)]))
        eval_a = (eva[0][self.filter(eva[1], normal)], np.zeros_like(eva[1][self.filter(eva[1], normal)]))
        eval_b = (eva[0][self.filter(eva[1], anomaly)], np.ones_like(eva[1][self.filter(eva[1], anomaly)]))
        # Stack both classes ontop of each other
        self.train = train_a
        self.eval = (np.concatenate((eval_a[0], eval_b[0], train_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1], train_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split anomaly MNIST"

class cifar10(dataset):
    """
    Class for the CIFAR-10 dataset.
    """

    def __init__(self):
        # Load CIFAR-10 data
        [train, eva] = tf.keras.datasets.cifar10.load_data()
        self.train = (train[0], np.squeeze(train[1]))
        self.eval = (eva[0], np.squeeze(eva[1].astype(np.uint8)))

    def get_dataset_name(self):
        return "CIFAR-10"

class split_anomaly_cifar10(cifar10):
    """
    Class for the CIFAR-10 dataset with task to classify two classes.
    """

    def __init__(self, normal = [0], anomaly = [1]):
        # Load CIFAR-10 data
        [train_tmp, eva_tmp] = tf.keras.datasets.cifar10.load_data()
        train = [train_tmp[0], np.squeeze(train_tmp[1])]
        eva = [eva_tmp[0], np.squeeze(eva_tmp[1].astype(np.uint8))]
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], normal)], np.zeros_like(train[1][self.filter(train[1], normal)]))
        train_b = (train[0][self.filter(train[1], anomaly)], np.ones_like(train[1][self.filter(train[1], anomaly)]))
        eval_a = (eva[0][self.filter(eva[1], normal)], np.zeros_like(eva[1][self.filter(eva[1], normal)]))
        eval_b = (eva[0][self.filter(eva[1], anomaly)], np.ones_like(eva[1][self.filter(eva[1], anomaly)]))
        # Stack both classes ontop of each other
        self.train = train_a
        self.eval = (np.concatenate((eval_a[0], eval_b[0], train_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1], train_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split anomaly CIFAR-10"

class landsat(dataset):
    """
    Class for the landsat sattelite image dataset.
    """

    def __init__(self):
        # Load landsat data
        train = np.loadtxt("../Datasets/UCI_Statlog_Landsat/sat.trn", delimiter=" ")
        eva = np.loadtxt("../Datasets/UCI_Statlog_Landsat/sat.tst", delimiter=" ")
        self.train = (train[:, 0:36], train[:, 36])
        self.eval = (eva[:, 0:36], eva[:, 36])

    def get_dataset_name(self):
        return "Landsat"

class split_anomaly_landsat(landsat):
    """
    Class for the landsat dataset with task to detect anomolies.
    """

    def __init__(self, normal = [0], anomaly = [1]):
        # Load landsat data
        train = np.loadtxt("../Datasets/UCI_Statlog_Landsat/sat.trn", delimiter=" ")
        eva = np.loadtxt("../Datasets/UCI_Statlog_Landsat/sat.tst", delimiter=" ")
        train = (train[:, 0:36], train[:, 36])
        eva = (eva[:, 0:36], eva[:, 36])
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], normal)], np.zeros_like(train[1][self.filter(train[1], normal)]))
        train_b = (train[0][self.filter(train[1], anomaly)], np.ones_like(train[1][self.filter(train[1], anomaly)]))
        eval_a = (eva[0][self.filter(eva[1], normal)], np.zeros_like(eva[1][self.filter(eva[1], normal)]))
        eval_b = (eva[0][self.filter(eva[1], anomaly)], np.ones_like(eva[1][self.filter(eva[1], anomaly)]))
        # Stack both classes ontop of each other
        self.train = train_a
        self.eval = (np.concatenate((eval_a[0], eval_b[0], train_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1], train_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split anomaly Landsat"


class kddcup99(dataset):
    """
    Class for the kddcup99 intrusion detection dataset.
    """

    def converter(self, s):
        if (s == b'icmp'): return 0.0
        elif (s == b'tcp'): return 1.0
        elif (s == b'udp'): return 2.0
        elif (s == b'IRC'): return 0.0
        elif (s == b'X11'): return 1.0
        elif (s == b'Z39_50'): return 2.0
        elif (s == b'aol'): return 3.0
        elif (s == b'atuh'): return 4.0
        elif (s == b'bgp'): return 5.0
        elif (s == b'courier'): return 6.0
        elif (s == b'csnet_ns'): return 7.0
        elif (s == b'ctf'): return 8.0
        elif (s == b'daytime'): return 9.0
        elif (s == b'discard'): return 10.0
        elif (s == b'domain'): return 11.0
        elif (s == b'domain_u'): return 12.0
        elif (s == b'echo'): return 13.0
        elif (s == b'eco_i'): return 14.0
        elif (s == b'ecr_i'): return 15.0
        elif (s == b'efs'): return 16.0
        elif (s == b'exec'): return 17.0
        elif (s == b'finger'): return 18.0
        elif (s == b'ftp'): return 19.0
        elif (s == b'ftp_data'): return 20.0
        elif (s == b'gopher'): return 21.0
        elif (s == b'harvest'): return 22.0
        elif (s == b'hostname'): return 23.0
        elif (s == b'http'): return 24.0
        elif (s == b'http_278'): return 25.0
        elif (s == b'http_443'): return 26.0
        elif (s == b'http_800'): return 27.0
        elif (s == b'imap4'): return 28.0
        elif (s == b'iso_tsap'): return 29.0
        elif (s == b'klogin'): return 30.0
        elif (s == b'kshell'): return 31.0
        elif (s == b'ldap'): return 32.0
        elif (s == b'link'): return 33.0
        elif (s == b'login'): return 34.0
        elif (s == b'mtp'): return 35.0
        elif (s == b'name'): return 36.0
        elif (s == b'netbios_'): return 37.0
        elif (s == b'netstat'): return 38.0
        elif (s == b'nnsp'): return 39.0
        elif (s == b'nntp'): return 40.0
        elif (s == b'ntp_u'): return 41.0
        elif (s == b'other'): return 42.0
        elif (s == b'pm_dump'): return 43.0
        elif (s == b'pop_2'): return 44.0
        elif (s == b'pop_3'): return 45.0
        elif (s == b'printer'): return 46.0
        elif (s == b'private'): return 47.0
        elif (s == b'red_i'): return 48.0
        elif (s == b'remote_j'): return 49.0
        elif (s == b'rje'): return 50.0
        elif (s == b'shell'): return 51.0
        elif (s == b'smtp'): return 52.0
        elif (s == b'sql_net'): return 53.0
        elif (s == b'ssh'): return 54.0
        elif (s == b'sunrpc'): return 55.0
        elif (s == b'supdup'): return 56.0
        elif (s == b'systat'): return 57.0
        elif (s == b'telnet'): return 58.0
        elif (s == b'tftp_u'): return 59.0
        elif (s == b'tim_i'): return 60.0
        elif (s == b'time'): return 61.0
        elif (s == b'urh_i'): return 62.0
        elif (s == b'urp_i'): return 63.0
        elif (s == b'uucp'): return 64.0
        elif (s == b'uucp_pat'): return 65.0
        elif (s == b'vmnet'): return 66.0
        elif (s == b'whois'): return 67.0
        elif (s == b'OTH'): return 0.0
        elif (s == b'REJ'): return 1.0
        elif (s == b'RSTO'): return 2.0
        elif (s == b'RSTR'): return 3.0
        elif (s == b'S0'): return 4.0
        elif (s == b'S1'): return 5.0
        elif (s == b'S2'): return 6.0
        elif (s == b'S3'): return 7.0
        elif (s == b'SF'): return 8.0
        elif (s == b'Sh'): return 9.0
        elif (s == b'back.'): return 0.0
        elif (s == b'buffer_overflow.'): return 1.0
        elif (s == b'ftp_write.'): return 2.0
        elif (s == b'guess_passwd.'): return 3.0
        elif (s == b'imap.'): return 4.0
        elif (s == b'ipsweep.'): return 5.0
        elif (s == b'land.'): return 6.0
        elif (s == b'loadmodule.'): return 7.0
        elif (s == b'multihop.'): return 8.0
        elif (s == b'neptune.'): return 9.0
        elif (s == b'nmap.'): return 10.0
        elif (s == b'normal.'): return 11.0
        elif (s == b'perl.'): return 12.0
        elif (s == b'phf.'): return 13.0
        elif (s == b'pod.'): return 14.0
        elif (s == b'portsweep.'): return 15.0
        elif (s == b'rootkit.'): return 16.0
        elif (s == b'satan.'): return 17.0
        elif (s == b'smurf.'): return 18.0
        elif (s == b'spy.'): return 19.0
        elif (s == b'teardrop.'): return 20.0
        elif (s == b'warezclient.'): return 21.0
        elif (s == b'warezmaster.'): return 22.0
        else: return -1.0

    def __init__(self):
        # Load kddcup99 data
        # data = np.loadtxt("../Datasets/KDDcup99_data/kddcup.data.corrected", delimiter=",", converters={1: self.converter, 2: self.converter, 3: self.converter, 41: self.converter}, dtype=np.float32)
        # np.save("../Datasets/KDDcup99_data/data.npy", data)
        # data = np.loadtxt("../Datasets/KDDcup99_data/kddcup.data_10_percent_corrected", delimiter=",", converters={1: self.converter, 2: self.converter, 3: self.converter, 41: self.converter}, dtype=np.float32)
        # np.save("../Datasets/KDDcup99_data/data_10_percent.npy", data)
        data = np.load("/home/sri/Desktop/sri_model/dataset/data.npy")
        self.train = (data[494021:, 0:41], data[494021:, 41])
        self.eval = (data[:494021, 0:41], data[:494021, 41])

    def get_dataset_name(self):
        return "KDDcup99"

class split_anomaly_kddcup99(kddcup99):
    """
    Class for the landsat dataset with task to detect anomolies.
    """

    def __init__(self, normal = [0], anomaly = [1]):
        # Load kddcup99 data
        # data = np.loadtxt("../Datasets/KDDcup99_data/kddcup.data.corrected", delimiter=",", converters={1: self.converter, 2: self.converter, 3: self.converter, 41: self.converter}, dtype=np.float32)
        # np.save("../Datasets/KDDcup99_data/data.npy", data)
        # data = np.loadtxt("../Datasets/KDDcup99_data/kddcup.data_10_percent_corrected", delimiter=",", converters={1: self.converter, 2: self.converter, 3: self.converter, 41: self.converter}, dtype=np.float32)
        # np.save("../Datasets/KDDcup99_data/data_10_percent.npy", data)
        N_eva = 49402
        N_train = 494021
        data = np.load("/home/srinivas/Desktop/sri_model/dataset/data.npy")
        norm = np.amax(data, axis=0)
        for i in range(norm.shape[0]-1):
            if (norm[i] != 0.0):
                data[:, i] /= norm[i]
        norm = np.amax(data, axis=0)
        train = (data[N_eva:N_train, 0:41], data[N_eva:N_train, 41])
        eva = (data[:N_eva, 0:41], data[:N_eva, 41])
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], normal)], np.zeros_like(train[1][self.filter(train[1], normal)]))
        train_b = (train[0][self.filter(train[1], anomaly)], np.ones_like(train[1][self.filter(train[1], anomaly)]))
        eval_a = (eva[0][self.filter(eva[1], normal)], np.zeros_like(eva[1][self.filter(eva[1], normal)]))
        eval_b = (eva[0][self.filter(eva[1], anomaly)], np.ones_like(eva[1][self.filter(eva[1], anomaly)]))
        # Stack both classes ontop of each other
        self.train = train_a
        self.eval = (np.concatenate((eval_a[0], eval_b[0], train_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1], train_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split anomaly Landsat"
