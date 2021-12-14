"""Waymo open dataset dataloader."""
import copy
import csv
import glob
import json
import os
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from hashlib import md5
from typing import List, Union

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset

from waymo_open_dataset import dataset_pb2 as open_dataset


def get_upright_box(waymo_box):
    """Convert waymo box to upright box format and return the convered box."""
    xmin = waymo_box.center_x-waymo_box.length/2
    # xmax = waymo_box.center_x+waymo_box.length/2
    ymin = waymo_box.center_y-waymo_box.width/2
    # ymax = waymo_box.center_y+waymo_box.width/2

    return [xmin, ymin, waymo_box.length, waymo_box.width]


class WaymoClassification(VisionDataset):
    """
    WaymoClassification dataset.

    Converted from Waymo Open Dataset for classification use.
    """

    def __init__(self, root: str, sample_list_name: Union[str, List[str]],
                 sample_list_root: str = None, subsample_idxs: List = None,
                 transform: callable = None, target_transform: callable = None,
                 resize_res: int = 32, coco: bool = False,
                 merge_label: bool = True, segment_indices: List = None,
                 label_type: str = 'human', **kwargs):
        """
        Constructor.

        Args
            root (string): Directory containing with the classificaiton images.
            sample_list_name (Either str or list of strs): Name(s) of the
                samplelist csv file(s) to include in this dataset. The list is
                generated using the generate_sample_list method.
            sample_list_root(str, optional): Path to the sample_list csv file.
            If None, root is treated as sample_list_root. Default: None.
            subsample_idxs (list, optional): Indexes to select from the
                sample_list.
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            resize_res (int, optional): Image size to resize to.
            coco(boolean, optional): Specify to True when labels are generated
                by golden model trained with COCO dataset. Default: False.
            merge_label(boolean): Convert from COCO label format to Waymo
                format if specified. Only work when coco flag is True. Default:
                False.
            segment_indcies(list, optional): Select which segments to use. Use
                all segments if segment_idx is None. Default: None.

        """
        super(WaymoClassification, self).__init__(root)
        self.sample_list_root = sample_list_root if sample_list_root else root
        self.sample_list_name = sample_list_name
        if isinstance(self.sample_list_name, list):
            self.sample_list_path = [os.path.join(
                self.sample_list_root, f"{s}_labels.csv")
                for s in self.sample_list_name]
        else:
            self.sample_list_path = os.path.join(
                self.sample_list_root, f"{self.sample_list_name}_labels.csv")
        self.transform = transform
        self.subsample_idxs = subsample_idxs
        self.resize_res = resize_res
        self.label_type = label_type

        self.coco = coco
        self.merge_label = merge_label
        self.segment_indices = segment_indices
        if isinstance(self.sample_list_path, list):
            self.samples = pd.concat([pd.read_csv(p)
                                      for p in self.sample_list_path], axis=0)
        else:
            self.samples = pd.read_csv(self.sample_list_path)

        if self.segment_indices is not None and \
                isinstance(self.segment_indices, list):
            segment_names = self.samples['segment'].unique()
            segs2drop = [segment_names[i] for i in range(
                len(segment_names)) if i not in segment_indices]
            for seg in segs2drop:
                idx2drop = self.samples[self.samples['segment'] == seg].index
                self.samples.drop(idx2drop, inplace=True)

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete.')
        if coco:
            self.samples.drop(
                self.samples[self.samples['class'] == 5].index, inplace=True)
            self.samples.drop(
                self.samples[self.samples['class'] == 7].index, inplace=True)
            self.samples.drop(
                self.samples[self.samples['class'] == 9].index, inplace=True)
            self.samples.drop(
                self.samples[self.samples['class'] == 11].index, inplace=True)
            self.samples.drop(
                self.samples[self.samples['class'] > 14].index, inplace=True)
            self.samples.reset_index(drop=True, inplace=True)
            if merge_label:
                # merge coco labels into waymo labels
                types = copy.deepcopy(self.samples.loc[:, 'class'])
                self.samples.loc[types == 3, 'class'] = 0  # car
                self.samples.loc[types == 4, 'class'] = 0  # motorcycle
                self.samples.loc[types == 6, 'class'] = 0  # bus
                self.samples.loc[types == 8, 'class'] = 0  # truck
                self.samples.loc[types == 1, 'class'] = 1  # person
                self.samples.loc[types == 10, 'class'] = 2  # traffic light
                self.samples.loc[types == 12, 'class'] = 2  # street sign
                self.samples.loc[types == 13, 'class'] = 2  # stop sign
                self.samples.loc[types == 14, 'class'] = 2  # parking meter
                self.samples.loc[types == 2, 'class'] = 3  # bicyle
            else:
                self.samples.loc[types == 3, 'class'] = 0  # car
                self.samples.loc[types == 4, 'class'] = 1  # motorcycle
                self.samples.loc[types == 6, 'class'] = 2  # bus
                self.samples.loc[types == 8, 'class'] = 3  # truck
                self.samples.loc[types == 1, 'class'] = 4  # person
                self.samples.loc[types == 10, 'class'] = 5  # traffic light
                self.samples.loc[types == 12, 'class'] = 6  # street sign
                self.samples.loc[types == 13, 'class'] = 7  # stop sign
                self.samples.loc[types == 14, 'class'] = 8  # parking meter
                self.samples.loc[types == 2, 'class'] = 9  # bicyle
        else:
            # remove unknown
            # self.samples.loc[:, 'class'] -= 1
            pass
        if not coco and not self.samples[self.samples['class'] > 3].empty:
            raise RuntimeError('class labels not in [0, 3]!')
        elif coco and merge_label and \
                not self.samples[self.samples['class'] > 3].empty:
            raise RuntimeError('class labels not in [0, 3]!')

        self.update_idxs()

        if self.subsample_idxs is not None:
            self.samples = self.samples[self.samples["idx"].isin(
                self.subsample_idxs)]
            # TODO: (romilb): Check if the update_idxs actually makes sense
            # here - no, because subsampling again would cause problems
            # self.update_idxs()

        # Replace NaN with ''
        self.samples['camera name'] = self.samples['camera name'].fillna('')

    def update_idxs(self):
        self.samples["idx"] = pd.Series(
            range(0, len(self.samples["idx"]))).values
        self.samples.set_index("idx", inplace=True, drop=False)

    def get_samplelist_id(self):
        if isinstance(self.sample_list_name, list):
            return "_".join(self.sample_list_name)
        else:
            return self.sample_list_name

    def __getitem__(self, idx):
        """Get a sample.
        Args
            index (int): Index
        Returns
            tuple: (image, target) where target is a tuple of all target types
                if target_type is a list with more than one item.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples.iloc[idx, :]

        img_name = sample['image name']
        if self.label_type == 'human':
            target = sample['class']
        elif self.label_type == 'golden_label':
            target = sample['golden_label']
        else:
            raise RuntimeError(f"Unrecognized label_type: {self.label_type}")
        seg_name = sample['segment']
        cam_name = sample['camera name']
        img_path = os.path.join(self.root, seg_name, cam_name, img_name)
        if self.resize_res is not None:
            image = Image.open(img_path).resize(
                (self.resize_res, self.resize_res), resample=PIL.Image.BICUBIC)
        else:
            image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    def get_md5(self):
        return md5(','.join((self.samples['image name'].values)).encode('utf-8')).hexdigest()

    @property
    def y(self):
        return self.samples["class"].values

    def get_column(self, column_name):
        """Get the value of a column."""
        return self.samples[column_name]

    def get_indexes(self, class_filter_list=None):
        """Return Indexes of elements which belong in the class_filter_list."""
        if class_filter_list is None:
            idx_series = self.samples["idx"]
        else:
            idx_series = self.samples[
                self.samples["class"].isin(class_filter_list)]["idx"]
        return idx_series.values

    def get_targets(self, class_filter_list=None):
        '''
        Returns
            Targets of elements which belong in the class_filter_list
        '''
        if class_filter_list is None:
            targets = self.samples["class"]
        else:
            targets = self.samples[self.samples["class"].isin(
                class_filter_list)]["class"]
        return targets.values

    def get_filtered_dataset(self, data_idxs, label_type='human',
                             custom_transforms=None):
        """Subsample the dataset.

        Args
            data_idxs:
            custom_transforms:
        Returns:
            WaymoClassification Object

        """
        trsf = custom_transforms if custom_transforms is not None else \
            self.transform
        dataset = WaymoClassification(self.root, self.sample_list_name,
                                      sample_list_root=self.sample_list_root,
                                      subsample_idxs=data_idxs,
                                      transform=trsf,
                                      target_transform=self.target_transform,
                                      coco=self.coco,
                                      merge_label=self.merge_label,
                                      resize_res=self.resize_res,
                                      segment_indices=self.segment_indices,
                                      label_type=label_type)
        for column_name in self.samples.columns:
            if column_name not in dataset.samples.columns:
                mask = self.samples["idx"].isin(data_idxs)
                col2add = self.samples[mask][column_name]
                dataset.samples[column_name] = col2add
        return dataset

    def get_filtered_loader(self, data_idxs, custom_transforms=None, **kwargs):
        """Return a Pytorch dataloader with only samples in data_idxs.

        Args
            data_idxs
            custom_transforms:
            kwargs:

        """
        subset_dataset = self.get_filtered_dataset(
            data_idxs, custom_transforms=custom_transforms)
        return DataLoader(subset_dataset, pin_memory=False, **kwargs,)

    def get_merged_dataset(self, other_dataset):
        '''Merge this instance of a dataset with another dataset.
        Args
            other_dataset(WaymoClassification)
        Returns
            WaymoClassification object
        '''
        if other_dataset is None:
            return self
        assert isinstance(other_dataset, WaymoClassification), \
            "The other dataset is not Waymo dataset."
        union_idxs = np.union1d(self.samples["idx"].values,
                                other_dataset.samples["idx"].values)
        return WaymoClassification(
            self.root, self.sample_list_name, subsample_idxs=union_idxs,
            sample_list_root=self.sample_list_root, transform=self.transform,
            target_transform=self.target_transform, resize_res=self.resize_res,
            coco=self.coco, merge_label=self.merge_label,
            segment_indices=self.segment_indices, label_type=self.label_type)

    def concat_dataset(self, other_dataset):
        '''
        Concatenates the samples of this dataset with another dataset inplace.
        Args
            other_dataset(WaymoClassification)
        Returns
            WaymoClassification object
        '''
        if other_dataset is None:
            return self
        assert isinstance(other_dataset, WaymoClassification), \
            "The other dataset is not WaymoClassification dataset."
        self.samples = pd.concat([self.samples, other_dataset.samples], axis=0)
        self.update_idxs()

    def resample(self, new_num_samples):
        current_num_samples = len(self.samples)
        print("Resampling from {} to {} samples".format(
            current_num_samples, new_num_samples))
        if current_num_samples >= new_num_samples:
            # Subsample
            subsample_idxs = np.random.choice(
                self.samples["idx"].values, new_num_samples, replace=False)
            return self.get_filtered_dataset(subsample_idxs)
        else:
            # Supersample
            num_reps = int(new_num_samples / current_num_samples)
            # Multiply quotient
            new_samples = pd.concat([self.samples] * num_reps)
            # Add remainder
            remainder = new_num_samples - len(new_samples)
            # print("remainder: {}".format(remainder))
            # print(len(self.samples))
            new_samples = pd.concat(
                [new_samples, self.samples.iloc[0:remainder]])
            self.samples = new_samples
            self.update_idxs()
            return self

    def get_split_indices(self, target_segment_name, split_time=10):
        """Split samples with in a segment into train and test dataset."""
        samples = self.samples[self.samples['segment'] == target_segment_name]
        start_t = samples.iloc[0]['timestamp']
        tdiff = (samples['timestamp'] - start_t) / 1000000
        former_indices = samples[tdiff < split_time]['idx'].to_list()
        latter_indices = samples[tdiff >= split_time]['idx'].to_list()
        return former_indices, latter_indices

    def get_class_dist(self):
        """Return class distribution of all samples."""
        class_dist = []
        if self.merge_label:
            class_cnt = 4
        else:
            class_cnt = 10
        for i in range(class_cnt):
            mask = self.samples['class'] == i
            class_dist.append(len(self.samples[mask]))
        return class_dist

    def get_time_of_day(self):
        """Return the majority of weather of all samples."""
        return self.samples['time of day'].mode().values[0]

    def get_weather(self):
        """Return the majority of weather of all samples."""
        return self.samples['weather'].mode().values[0]

    @staticmethod
    def get_camera_resolution(cam_calibrations, cam_name):
        """Get camera resolution."""
        for cam_calibration in cam_calibrations:
            if cam_calibration.name == cam_name:
                return (cam_calibration.width, cam_calibration.height)
        return None

    @staticmethod
    def get_frame_labels(projected_lidar_labels, image):
        bboxes = []
        obj_ids = []
        obj_types = []
        detection_difficulty_levels = []
        tracking_difficulty_levels = []
        for labels in projected_lidar_labels:
            if labels.name != image.name:
                continue
            for obj_label in labels.labels:
                bboxes.append(get_upright_box(obj_label.box))
                obj_ids.append(obj_label.id)
                obj_types.append(obj_label.type)
                detection_difficulty_levels.append(
                    obj_label.detection_difficulty_level)
                tracking_difficulty_levels.append(
                    obj_label.tracking_difficulty_level)
        return obj_ids, bboxes, obj_types, detection_difficulty_levels, \
            tracking_difficulty_levels

    @staticmethod
    def generate_samples(segment, seg_name: str, writer,
                         seg_save_root: str = "",
                         min_res: int = 224, gt_file: str = "",
                         min_time_gap: float = 0.5):
        '''Generate sample ground truth and classification images from tfrecord
           or a groudn truth file.

        Args:
            segment: a segment tfrecord.
            seg_name: the name of the segment.
            writer:
            seg_save_root: the directory that holds all segment files.
            min_res: minimum resolution of cropped images
            gt_file: a ground truth file that holds bounding box information.
            min_time_gap: Minimum time gap between two frames from the same
            camera
        '''
        assert writer is not None
        sample_cnt = 0
        # Camera_name to last frame mapping for enforcing min_time_gap
        last_frame_timestamp = {}
        for frame_idx, data in enumerate(segment):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            for image in frame.images:
                # Min time gap check
                # FRONT, SIDE_LEFT etc.
                camera_name = open_dataset.CameraName.Name.Name(image.name)
                this_ts = frame.timestamp_micros
                last_ts = last_frame_timestamp.get(camera_name, 0)
                # if last frame was extracted within min_time_gap, move onto next frame
                # Micro sec conversion
                if this_ts - last_ts < min_time_gap * (10**6):
                    continue
                # if not, update last ts and run extraction
                last_frame_timestamp[camera_name] = this_ts

                if seg_save_root:
                    img = Image.fromarray(
                        np.array(tf.image.decode_jpeg(image.image)), 'RGB')
                resol = WaymoClassification.get_camera_resolution(
                    frame.context.camera_calibrations, image.name)
                if not gt_file:
                    obj_ids, bboxes, obj_types, detection_difficulty_levels, \
                        tracking_difficulty_levels = \
                        WaymoClassification.get_frame_labels(
                            frame.projected_lidar_labels, image)
                else:
                    if not os.path.exists(gt_file):
                        raise Exception(f'{gt_file} does not exist!')
                    gtruth = load_object_detection_results(gt_file, True)
                    bboxes = gtruth[frame_idx]
                    obj_types = [box[4] for box in bboxes]
                    obj_ids = list(range(len(bboxes)))
                    detection_difficulty_levels = [box[5] for box in bboxes]
                    tracking_difficulty_levels = [box[5] for box in bboxes]
                for obj_id, box, obj_type, detection_difficulty_level, \
                    tracking_difficulty_level in \
                    zip(obj_ids, bboxes, obj_types,
                        detection_difficulty_levels,
                        tracking_difficulty_levels):
                    # do not consider boxes that are too small
                    # if frcnn and obj_type not in {1, 2, 3, 4, 6,
                    #                               8, 10, 12, 13}:
                    #     # Check coco dataset for the labels
                    #     continue
                    if (box[2]*box[3] / (min_res**2)) < 0.5:
                        continue
                    if box[2] < min_res and box[3] < min_res:
                        box[0] -= (min_res - box[2]) / 2
                        box[1] -= (min_res - box[3]) / 2
                        box[2] = box[0] + min_res
                        box[3] = box[1] + min_res
                    elif box[2] < min_res:
                        # print(box[0], box[1], box[2], box[3])
                        if box[2] * min_res / (min_res ** 2) < 0.5:
                            continue
                        box[0] -= (min_res-box[2]) / 2
                        box[2] = box[0] + min_res
                        box[3] += box[1]
                    elif box[3] < min_res:
                        if box[3] * min_res / (min_res ** 2) < 0.5:
                            continue
                        # print(box[0], box[1], box[2], box[3])
                        box[1] -= (min_res - box[3]) / 2
                        box[2] += box[0]
                        box[3] = box[1] + min_res
                    else:
                        box[2] += box[0]
                        box[3] += box[1]
                    if box[0] < 0:
                        box[0] = 0
                        box[2] = min_res
                    if box[1] < 0:
                        box[1] = 0
                        box[3] = min_res
                    if box[2] > resol[0]:
                        box[2] = resol[0]
                        box[0] = box[2] - min_res
                    if box[3] > resol[1]:
                        box[3] = resol[1]
                        box[1] = box[3] - min_res
                    if gt_file:
                        img_name = '{:04d}_{}_{}.jpg'.format(
                            frame_idx, obj_id,
                            open_dataset.CameraName.Name.Name(image.name))
                    else:
                        img_name = '{:04d}_{}.jpg'.format(frame_idx, obj_id)
                    writer.writerow([sample_cnt, img_name, obj_type,
                                     frame.timestamp_micros,
                                     frame.context.stats.weather,
                                     frame.context.stats.location,
                                     frame.context.stats.time_of_day,
                                     seg_name, frame_idx,
                                     open_dataset.CameraName.Name.Name(
                                         image.name),
                                     obj_id, box[0], box[1], box[2], box[3],
                                     detection_difficulty_level,
                                     tracking_difficulty_level])
                    sample_cnt += 1
                    if seg_save_root:
                        cam_save_root = os.path.join(
                            seg_save_root, open_dataset.CameraName.Name.Name(
                                image.name))
                        if not os.path.exists(cam_save_root):
                            os.mkdir(cam_save_root)
                        img_path = os.path.join(cam_save_root, img_name)
                        cropped_img = img.crop(
                            tuple(box[:4])).resize((min_res, min_res))
                        cropped_img.save(img_path)
                        cropped_img.close()
                if seg_save_root:
                    img.close()

    @staticmethod
    def generate_sample_list(
            sorted_segs_file: str, city: str, img_save_path: str = "",
            write_filename: str = "", min_res: int = 224, date_range=None,
            start_seg_index=0, num_segs=-1, gt_file: str ="", min_time_gap=0.5):
        """Generate sample list."""
        with open(sorted_segs_file, 'r') as f:
            sorted_segs = json.load(f)

        if write_filename:
            fout = open(write_filename, "w", 1)
            writer = csv.writer(fout)
            writer.writerow(['idx', 'image name', 'class', 'timestamp',
                             'weather', 'location', 'time of day', 'segment',
                             'frame index', 'camera name', 'label id',
                             'xmin', 'ymin', 'xmax', 'ymax',
                             'detection difficulty level',
                             'tracking difficulty level'])
        else:
            writer = None
        if num_segs == -1:
            num_segs = len(sorted_segs[f'location_{city}'])
        seg_files = list()
        for i, pair in enumerate(sorted_segs[f'location_{city}']):
            if i < start_seg_index or i >= start_seg_index+num_segs:
                continue
            seg_ts = datetime.utcfromtimestamp(pair[1]/1000000)
            tfrec_path = pair[0]
            if date_range is None:
                seg_files.append(tfrec_path)
            elif date_range[0] <= seg_ts <= date_range[1]:
                seg_files.append(tfrec_path)

        print('Num of seg files', len(seg_files))

        for seg_file in seg_files:
            segment = tf.data.TFRecordDataset(seg_file, compression_type='')
            seg_name = os.path.splitext(os.path.basename(seg_file))[0]
            if img_save_path:
                seg_img_save_path = os.path.join(img_save_path, seg_name)
                if not os.path.exists(seg_img_save_path):
                    os.mkdir(seg_img_save_path)
            else:
                seg_img_save_path = ""

            start_t = time.time()
            WaymoClassification.generate_samples(
                segment, seg_name, writer, seg_img_save_path, min_res,
                gt_file=gt_file, min_time_gap=min_time_gap)
            print("Processing {} cost {:.2f} seconds".format(
                seg_file, time.time()-start_t))
        if writer is not None:
            fout.close()

    @staticmethod
    def sort_segments(root: str, save_dir: str):
        """
        Sort segments based on the timestamp of the 1st frame.

        Args
            root: directory where *.tfrecord segment files exist
            save_file: save a json file if specified

        Returns
            segments(dictionary): {location: [[segments, timestamp], ...]
        """
        ret_segs = defaultdict(list)
        timestamps = defaultdict(list)
        segs = glob.glob(os.path.join(root, '*.tfrecord'))
        for seg_file in segs:
            segment = tf.data.TFRecordDataset(seg_file, compression_type='')
            for _, data in enumerate(segment):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                ret_segs[frame.context.stats.location].append(seg_file)
                timestamps[frame.context.stats.location].append(
                    frame.timestamp_micros)
                break  # only need 1st frame to check the timestamp
        for loc in ret_segs:
            segs = ret_segs[loc]
            loc_ts = timestamps[loc]
            ret_segs[loc] = sorted(zip(segs, loc_ts), key=lambda pair: pair[1])

        with open(os.path.join(save_dir, 'sorted_segments.json'), 'w') as fout:
            json.dump(ret_segs, fout, sort_keys=True, indent=4)
        return ret_segs


def load_object_detection_results(filename, width_height=False):
    """Load object detection results.

    The csv file should contain 8 columns
    (frame id, xmin, ymin, xmax, ymax, t, score, object id)
    or 7 columns
    (frame id, xmin, ymin, xmax, ymax, t, score).

    Args
        filename(string): filename of object detection results in csv format.
        width_height(bool): Return box format [xmin, ymin, w, h, ...] if True.
                            Otherwise, return [xmin, ymin, xmax, ymax, ...].
                            Default: False.
    Return
        dets(dict): a dict mapping frame id to bounding boxes.
    """
    dets = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            if len(row) == 8:
                frame_id, xmin, ymin, xmax, ymax, t, score, obj_id = row
                if int(frame_id) not in dets:
                    dets[int(frame_id)] = []
                if xmin and ymin and xmax and ymax and t and score and obj_id:
                    if width_height:
                        dets[int(frame_id)].append([
                            float(xmin), float(ymin), float(xmax)-float(xmin),
                            float(ymax)-float(ymin), int(float(t)),
                            float(score), int(float(obj_id))])
                    else:
                        dets[int(frame_id)].append([
                            float(xmin), float(ymin), float(xmax), float(ymax),
                            int(float(t)), float(score), int(float(obj_id))])
            elif len(row) == 7:
                frame_id, xmin, ymin, xmax, ymax, t, score = row
                if int(frame_id) not in dets:
                    dets[int(frame_id)] = []
                if xmin and ymin and xmax and ymax and t and score:
                    if width_height:
                        dets[int(frame_id)].append([
                            float(xmin), float(ymin), float(xmax)-float(xmin),
                            float(ymax)-float(ymin), int(t), float(score)])
                    else:
                        dets[int(frame_id)].append([
                            float(xmin), float(ymin), float(xmax), float(ymax),
                            int(t), float(score)])

    return dets
