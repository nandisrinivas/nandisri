"""
This script is used to create average metrics.
"""

import numpy as np
import matplotlib.pyplot as plt

paths = ["./log/EWC/2018_10_09_16_17",\
        "./log/EWC/2018_10_09_16_37",\
        "./log/EWC/2018_10_09_17_05",\
        "./log/EWC/2018_10_09_17_43",\
        "./log/EWC/2018_10_09_18_30",\
        "./log/EWC/2018_10_09_19_27",\
        "./log/EWC/2018_10_09_20_36",\
        "./log/EWC/2018_10_09_21_55",\
        "./log/EWC/2018_10_09_23_25",\
        "./log/EWC/2018_10_10_01_07"]

metric_list = []
for p in paths:
    metrics = np.load(p+"/metrics.npy")
    metrics = np.transpose(np.reshape(metrics, [16, 8]))
    metric_list.append(metrics)

avg_metrics = np.zeros([8, 16])
for m in metric_list:
    avg_metrics += m
avg_metrics /= len(paths)

metric_names = ["acc", "auc", "prec", "rec", "tp", "fp", "tn", "fn"]
for i, m_n in enumerate(metric_names):
    y = avg_metrics[i, :]
    x = np.arange(1,y.shape[0]+1)
    data = np.transpose(np.vstack((x, y)))
    np.savetxt("./log/EWC/"+m_n+".csv", data, delimiter=",", header="a,b", comments="")
