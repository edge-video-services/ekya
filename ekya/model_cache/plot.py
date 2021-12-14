import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ekya.utils.helpers import read_json_file

ASPECT = 1.3
HEIGHT = 2.5
PARAMS = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}

plt.rcParams.update(PARAMS)

ROOT = '/data2/zxxia/ekya/ekya/model_cache/results_golden_label'
CITIES = ['sf_030_039', 'sf_050_059', 'sf_060_069',
          'sf_070_079', 'sf_080_089']
# CITIES = ['sf_060_069']

def select_by_location(test_data, mask):
    """Select model by their locaiton. Due to lack of gps signal, we select a
    model from a city to mimic this baseline."""
    return test_data[mask].iloc[3]['test_acc']


def select_by_obj_cnt(test_data, mask, obj_cnt):
    """Select model by number of objects in a window."""
    obj_cnt_diff = np.inf
    selected_row = None
    for row_idx, row in test_data.loc[mask].iterrows():
        cached_metadata = read_json_file(os.path.join(
            'tmp', row['cached_camera_name'],
            "{}_metadata.json".format(row['hyp_id'])))
        candi_cls_dist = cached_metadata[str(
            row['task_id'])]['test']['class_distribution']
        candi_obj_cnt = np.sum(candi_cls_dist)
        # candi_dist = np.linalg.norm(np.array(cls_dist) - np.array(candi_cls_dist))
        if np.abs(candi_obj_cnt - obj_cnt) < obj_cnt_diff:
            selected_row = row
            obj_cnt_diff = np.abs(candi_obj_cnt - obj_cnt)
        return row['test_acc']


def select_by_tod(test_data, mask, tod):
    selected_tod_accs = []
    selected_tod = []
    selected_tod_task_id = []
    for row_idx, row in test_data.loc[mask].iterrows():
        cached_metadata = read_json_file(os.path.join(
            'tmp', row['cached_camera_name'],
            "{}_metadata.json".format(row['hyp_id'])))
        if cached_metadata[str(row['cached_camera_task_id'])]['test']['time_of_day'] == tod:
            selected_tod_accs.append(row['test_acc'])
            return row['test_acc']


waymo_ekya_accs = []
waymo_selected_accs = []
waymo_selected_tod_accs = []
waymo_selected_obj_cnt_accs = []
waymo_selected_location_accs = []

for city in CITIES:
    test_data = pd.read_csv(os.path.join(ROOT, '{}.csv'.format(city)))
    ekya_accs = []
    ekya_model_names = []
    cached_model_accs_mean = []
    cached_model_accs_max = []
    cached_model_accs_min = []
    selected_accs = []
    selected_tod_accs = []
    selected_obj_cnt_accs = []
    selected_location_accs = []
    metadata = read_json_file(os.path.join(
        'tmp', city, "{}_metadata.json".format(0)))
    for task_id in range(1, 10):
        # mask = test_data.loc[:, 'task_id'] == task_id
        cls_dist = metadata[str(task_id)]['test']['class_distribution']
        obj_cnt = np.sum(cls_dist)
        tod = metadata[str(task_id)]['test']['time_of_day']
        mask = ((test_data['task_id'] == task_id) &
                (test_data['cached_camera_task_id'] == task_id) &
                (test_data['cached_camera_name'] == city))
        ekya_accs.append(test_data.loc[mask, 'test_acc'].max())
        max_idx = test_data.loc[mask, 'test_acc'].argmax()
        ekya_model_names.append(
            test_data[mask].iloc[max_idx]['cached_camera_name'])
        mask = (test_data['task_id'] == task_id) #& (
            #test_data['cached_camera_name'].isin(["sf_000_009", "sf_020_029"]))
        # location_mask = (test_data['task_id'] == task_id) & (
        #     test_data['cached_camera_name']== city) # deprecated

        distance = np.inf
        best_row = None
        for row_idx, row in test_data.loc[mask].iterrows():
            cached_metadata = read_json_file(os.path.join(
                'tmp', row['cached_camera_name'],
                "{}_metadata.json".format(row['hyp_id'])))
            candi_cls_dist = cached_metadata[str(
                row['task_id'])]['test']['class_distribution']
            candi_dist = np.linalg.norm(
                np.array(cls_dist) - np.array(candi_cls_dist))
            if candi_dist < distance:
                best_row = row

        cached_model_accs_mean.append(test_data.loc[mask, 'test_acc'].mean())
        cached_model_accs_max.append(test_data.loc[mask, 'test_acc'].max())
        cached_model_accs_min.append(test_data.loc[mask, 'test_acc'].min())
        if best_row is not None:
            selected_accs.append(best_row['test_acc'])
        else:
            selected_accs.append(np.nan)
        selected_tod_accs.append(select_by_tod(test_data, mask, tod))
        selected_obj_cnt_accs.append(
            select_by_obj_cnt(test_data, mask, obj_cnt))
        selected_location_accs.append(select_by_location(test_data, mask))

    # print(ekya_model_names)
    waymo_ekya_accs.extend(ekya_accs)
    waymo_selected_accs.extend(selected_accs)
    waymo_selected_tod_accs.extend(selected_tod_accs)
    waymo_selected_obj_cnt_accs.extend(selected_obj_cnt_accs)
    waymo_selected_location_accs.extend(selected_location_accs)
    if city != 'sf_060_069':
        continue
    plt.figure(figsize=(ASPECT*HEIGHT*1.3, HEIGHT*1.4))
    ax = plt.gca()
    marker_size = 6
    line1, = ax.plot(range(1, 10), selected_accs, 'o-', c='C0', ms=marker_size,
                     label="Class distribution based model selection")
    line2, = ax.plot(range(1, 10), selected_tod_accs, 'o-', c='C1', ms=marker_size,
                     label="Time of day based model selection")
    line3, = ax.plot(range(1, 10), selected_obj_cnt_accs, 'o-', c='C3', ms=marker_size,
                     label="Object count based model selection")
    line4, = ax.plot(range(1, 10), selected_location_accs, 'o-', c='C4', ms=marker_size,
                     label="Location based model selection")
    line0, = ax.plot(range(1, 10), ekya_accs, 'o-',
                     c='C2', ms=marker_size, label="Ekya")
    ax.set_xticks(np.arange(1, 10))
    ax.set_xticklabels([str(x) for x in np.arange(1, 10)])
    ax.set_xlabel('Retraining window(over time)')
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0., 1.0)
    ax.set_xlim(0.5, 9.5)
    ax.legend((line0, line1, line2, line3, line4),
              ("Ekya", "Time of day based", "Class distribution based",
               "Object count based", "Location based"), loc="lower right",
               ncol=1, handletextpad=0.2,
              borderaxespad=0.15,
              borderpad=0.2,
              handlelength=0.8,
              columnspacing=0.1,
              labelspacing=0.1,
              handleheight=0.8, fontsize=11.5)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"waymo_{city}_golden_label.pdf")


plt.figure(figsize=(ASPECT*HEIGHT*1.3, HEIGHT * 1.4))
ax = plt.gca()

cls_dist_acc_diff = -np.array(waymo_selected_accs) + np.array(waymo_ekya_accs)
tod_acc_diff = -np.array(waymo_selected_tod_accs) + np.array(waymo_ekya_accs)
obj_cnt_acc_diff = -np.array(waymo_selected_obj_cnt_accs) + np.array(waymo_ekya_accs)
location_acc_diff = -np.array(waymo_selected_location_accs) + np.array(waymo_ekya_accs)
# cls_dist_err = np.std(-np.array(waymo_selected_accs) + np.array(waymo_ekya_accs))/ np.sqrt(len(waymo_ekya_accs))# / np.array(waymo_selected_accs))
# tod_err = np.std(-np.array(waymo_selected_tod_accs) + np.array(waymo_ekya_accs))/ np.sqrt(len(waymo_ekya_accs))# / np.array(waymo_selected_tod_accs))
# obj_cnt_err = np.std(-np.array(waymo_selected_obj_cnt_accs) + np.array(waymo_ekya_accs))/ np.sqrt(len(waymo_ekya_accs))# / np.array(waymo_selected_obj_cnt_accs))
# location_err = np.std(-np.array(waymo_selected_location_accs) + np.array(waymo_ekya_accs))/ np.sqrt(len(waymo_ekya_accs))# / np.array(waymo_selected_location_accs))
bp = ax.boxplot([cls_dist_acc_diff, tod_acc_diff, obj_cnt_acc_diff, location_acc_diff], notch=True, patch_artist=True)
colors = ['C0', 'C1', 'C3', 'C4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for whisker in bp['whiskers']:
    whisker.set(linestyle =":")

for median in bp['medians']:
    median.set(color ='yellow', linewidth = 1.5)
ax.tick_params(axis='x', which='major', pad=0)
ax.set_xticks(np.arange(1, 5))
ax.set_xticklabels(["Class\ndistribution\nbased", "Time of day\nbased",
                    "Object count\nbased", "Location\nbased"], rotation=45,
                    linespacing=0.8)
ax.yaxis.set_label_coords(x=-0.2, y=0.4)
ax.set_ylabel("Ekya’s acc - Baseline's acc")
ax.grid(axis='y')
plt.tight_layout()
# ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 0.8)
# plt.show()
plt.savefig("waymo_reuse_cached_model_boxplot.pdf")
