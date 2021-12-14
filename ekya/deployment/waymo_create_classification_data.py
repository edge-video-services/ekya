"""Crop classification image from the video frames."""
import os

import tensorflow as tf
from ekya.datasets.WaymoClassification import WaymoClassification

WAYMO_ROOT = '/home/romilb/datasets/waymo'
SEGMENTS_PATH = '/home/romilb/datasets/waymo/sorted_segments.json'
SAVE_PATH = '/home/romilb/datasets/waymo/waymo_classification_images2'
CITY = 'phx'

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')
    # use object detection results
    # WaymoClassification.generate_sample_list(
    #     './sorted_segments.json', 'phx',
    #     img_save_path='/data/zxxia/ekya/datasets/waymo_classification_images_nas',
    #     write_filename='/data/zxxia/ekya/datasets/waymo_classification_images_nas/sf_labels.csv',
    #     model_name='faster_rcnn_nas')

    # use waymo ground truth
    WaymoClassification.generate_sample_list(
        sorted_segs_file=SEGMENTS_PATH,
        city=CITY,
        waymo_root=WAYMO_ROOT,
        img_save_path=SAVE_PATH,
        write_filename=os.path.join(SAVE_PATH, f'{CITY}_labels.csv'),
        start_seg_index=20,
        num_segs=40,
        min_time_gap=1)


if __name__ == "__main__":
    main()
