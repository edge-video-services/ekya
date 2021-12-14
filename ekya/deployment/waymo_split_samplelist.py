"""Split a large sample list into many short sample lists.
"""
import copy
import json
import os

import pandas as pd

ROOT = '/home/romilb/datasets/waymo/waymo_classification_images/sample_lists/citywise/'
CITY = 'phx'
NUM_SEGS = 10
seg_offset = 20 # Offset wrt the original sorted segment list file. Used for naming consistency.
sample_list_file = os.path.join(ROOT, f'{CITY}_labels.csv')
samples = pd.read_csv(sample_list_file)
sorted_segs = samples['segment'].unique()

seg_win = []
for i, seg in enumerate(sorted_segs):
    if (i + 1) % NUM_SEGS == 0:
        print(i, i - NUM_SEGS + 1, seg_win)
        sample2save = copy.deepcopy(samples[samples['segment'].isin(seg_win)])
        sample2save["idx"] = pd.Series(
            range(0, len(sample2save["idx"]))).values
        sample2save.set_index("idx", inplace=True, drop=True)
        print(sample2save['segment'].unique())
        outfile = os.path.join(ROOT, f'{CITY}_{seg_offset+i-NUM_SEGS+1}_{seg_offset+i}_labels.csv')
        sample2save.to_csv(outfile)

        seg_win = []
    seg_win.append(seg)