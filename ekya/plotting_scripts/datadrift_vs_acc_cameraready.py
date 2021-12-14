"""
Plotting accuracy drift vs change in data distribution
Adopted from https://github.com/romilbhardwaj/ekya/blob/zxxia_core/plotting_scripts/motivation_plot.py
"""
import copy
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def l2_norm(a,b):
    return np.linalg.norm(a-b)

def chi2(A, B):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(A, B)])
    return chi

from config import ASPECT, HEIGHT, PARAMS

OUTPUT_PATH = './output/'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
plt.rcParams.update(PARAMS)
DATASET = 'cityscapes'
# DATASET = 'waymo'
if DATASET == 'cityscapes':
    CITIES = ['aachen', "bochum", "bremen", "cologne", "darmstadt",
              "dusseldorf", 'jena', "monchengladbach", "stuttgart", "tubingen",
              "zurich"]
    # ["aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",
    # "erfurt", "frankfurt", "hamburg", "hanover", "jena", "krefeld", "lindau",
    # "monchengladbach", "munster", "strasbourg", "stuttgart", "tubingen",
    # "ulm", "weimar", "zurich"]
elif DATASET == 'waymo':
    raise NotImplementedError

MODEL_NAME = 'resnet18'
NUM_TASKS = 10
NUM_TASKS_TO_TRAIN = 5
NUM_HIDDEN = 64

city = 'zurich'
height = 3.3
aspect = 1.4
plt.figure(figsize=[aspect*height, height])
plt.rcParams.update(PARAMS)
ax = plt.gca()

# df = pd.read_csv(os.path.join(
#     ROOT, DATASET, f'cityscapes_{city}_resnet18_64_10tasks_train5tasks_retrain_True_test_accs.csv'))
# ax.plot(df.iloc[5:]['task id']+1, df.iloc[5:]['retrain test accs'],
#         marker='o', c='#1f77b4', label='continunously retrained model')


# plot citysacpes motivation class distribution figure
column_names = ["idx", "imgpath", "class", "x0", "y0", "x1", "y1"]
df = pd.read_csv(
    f'/mnt/d/RomilOffline/research/datasets/cityscapes/sample_lists/citywise/{city}_fine.csv',
    names=column_names)
num_classes = 6
num_tasks = 10
num_samples_per_task = int(len(df) / num_tasks)
class_cnts = []
for task_id in range(num_tasks):
    class_cnt = np.zeros(num_classes)
    samples = df.iloc[task_id *
                      num_samples_per_task:(task_id+1) * num_samples_per_task]
    classes = samples['class'].to_list()
    cnt = Counter(classes)
    for label in range(num_classes):
        class_cnt[label] += cnt[label]
    class_cnts.append(class_cnt)
class_cnts = np.array(class_cnts)
print(ASPECT, HEIGHT)
class_cnts_1_5 = (
    np.sum(class_cnts[0:5], axis=0) / (num_samples_per_task * 5)).reshape(1, -1)
class_cnts = class_cnts[5:] / num_samples_per_task

class_cnts = np.concatenate([class_cnts_1_5, class_cnts], axis=0)
fig = plt.figure(figsize=[aspect*height, height])
plt.rcParams.update(PARAMS)
ax = fig.gca()
label_vals = [4, 3, 1, 5, 0, 2]
labels = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']

for idx, (i, label) in enumerate(zip(label_vals, labels)):
    ax.bar(range(5, num_tasks+1), class_cnts[:, i], width=0.5,
           bottom=np.sum(class_cnts[:, label_vals[0:idx]], axis=1),
           label=label)

print(class_cnts)
ax.set_xlabel('Retraining Window (over time)')
ax.set_ylabel('Class Frequency')
ax.set_xticks(range(5, num_tasks+1))
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xticklabels(['1-5', '6', '7', '8', '9', '10'])

ax.set_ylim(0, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(-0, 1.02, 1, 0.2), ncol=3,
          prop={'size': 'large'})
fig.tight_layout()
# plt.savefig(
#     os.path.join(OUTPUT_PATH, f'motivation_cityscapes_{city}_classdist.pdf'),
#     bbox_inches='tight')

dist_fn = cos_sim
print(dist_fn(class_cnts[0], class_cnts[1]))
print(dist_fn(class_cnts[0], class_cnts[2]))
print(dist_fn(class_cnts[0], class_cnts[3]))
print(dist_fn(class_cnts[0], class_cnts[4]))
print(dist_fn(class_cnts[0], class_cnts[5]))
print(len(class_cnts))

# Read data
df = pd.read_csv('datadrift.csv')
u, df["label_num"] = np.unique(df["city"], return_inverse=True)
print(df)
df['accdrift']*=-1
CITIES = pd.unique(df['city'])

print(CITIES)

plt.rcParams.update(PARAMS)
plt.figure(figsize=[aspect*height, height])
ax = plt.gca()

# df.plot.scatter(x='similarity',y='accdrift', ax=ax)
#
# sc = ax.scatter(x="similarity", y="accdrift", c="label_num", data=df)
# ax.legend(sc.legend_elements()[0], u, title="City")

markers = {
    'zurich': 'o',
    'aachen': 's',
    'jena': '*',
    'bremen': 'D',
    'bochum': '^'
}

labels = {
    'zurich': 'C1',
    'aachen': 'C2',
    'jena': 'C3',
    'bremen': 'C4',
    'bochum': 'C5'
}

for city in CITIES:
    temp_df = df[df['city'] == city]
    ax.plot(list(temp_df['similarity']), list(temp_df['accdrift']),
            ls='--',
            alpha=0.5)
    ax.scatter(list(temp_df['similarity']), list(temp_df['accdrift']),
            label=labels[city],
            marker=markers[city],
            s=64,
            alpha=1)


plt.legend(bbox_to_anchor=(-0.2, 1.05, 1, 0.2), ncol=5, borderaxespad=0.1, handletextpad=0.1, columnspacing=0.3, loc="upper left")
plt.grid()

x_tick_range = np.arange(0.9, 1.1, 0.02)
ax.set_xticks(x_tick_range)
plt.xlim([0.9, 1])

y_tick_range = np.arange(-0.3, 0.001, 0.05)
ax.set_yticks(y_tick_range)
plt.ylim([-0.31, 0])

plt.ylabel('Accuracy Difference')
plt.xlabel('Class Distribution Cosine Similarity')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'motivation_datadrift_vs_acc.pdf', bbox_inches='tight')
