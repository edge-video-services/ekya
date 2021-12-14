"""Wrapper of Mp4Video video."""
import csv
import copy
import os
import glob
import subprocess
from hashlib import md5
from typing import List, Union

import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class Mp4VideoClassification(VisionDataset):
    def __init__(self,
                 root: str,
                 sample_list_name: Union[str, List[str]],
                 sample_list_root: str = None,
                 subsample_idxs: List = None,
                 transform: callable = None,
                 target_transform: callable = None,
                 use_cache: bool = False,
                 resize_res: int = 32,
                 label_type: str = 'human'):    # 'golden_model'
        super(Mp4VideoClassification, self).__init__(root)
        self.resize_res = resize_res
        self.use_cache = use_cache
        self.sample_list_name = sample_list_name
        self.sample_list_root = sample_list_root if sample_list_root else root
        if isinstance(self.sample_list_name, list):
            self.sample_list_path = [os.path.join(
                self.sample_list_root, f"{s}.csv") for s in
                self.sample_list_name]
        else:
            self.sample_list_path = os.path.join(
                self.sample_list_root, f"{self.sample_list_name}.csv")

        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.subsample_idxs = subsample_idxs
        self.label_type = label_type

        if isinstance(self.sample_list_path, list):
            self.samples = pd.concat([pd.read_csv(p)
                                      for p in self.sample_list_path], axis=0)
        else:
            self.samples = pd.read_csv(self.sample_list_path)

        if self.subsample_idxs is not None:
            self.samples = self.samples[self.samples["idx"].isin(
                self.subsample_idxs)]
        self.samples.reset_index(drop=True, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] == 5].index, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] == 7].index, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] == 9].index, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] == 10].index, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] == 11].index, inplace=True)
        self.samples.drop(
            self.samples[self.samples['class'] >= 13].index, inplace=True)
        self.samples.reset_index(drop=True, inplace=True)
        types = copy.deepcopy(self.samples.loc[:, 'class'])

        # merge coco labels into waymo labels
        # self.samples.loc[types == 3, 'class'] = 0  # car
        # self.samples.loc[types == 4, 'class'] = 0  # motorcycle
        # self.samples.loc[types == 6, 'class'] = 0  # bus
        # self.samples.loc[types == 8, 'class'] = 0  # truck
        # self.samples.loc[types == 1, 'class'] = 1  # person
        # self.samples.loc[types == 10, 'class'] = 2  # traffic light
        # self.samples.loc[types == 12, 'class'] = 2  # street sign
        # self.samples.loc[types == 13, 'class'] = 2  # stop sign
        # self.samples.loc[types == 14, 'class'] = 2  # parking meter
        # self.samples.loc[types == 2, 'class'] = 3  # bicyle

        # merge coco labels into cityscapes labels
        self.samples.loc[types == 1, 'class'] = 0  # person
        self.samples.loc[types == 3, 'class'] = 1  # car
        self.samples.loc[types == 8, 'class'] = 2  # truck
        self.samples.loc[types == 6, 'class'] = 3  # bus
        self.samples.loc[types == 2, 'class'] = 4  # bicyle
        self.samples.loc[types == 4, 'class'] = 5  # motorcycle

        # reorganize coco labels
        # self.samples.loc[types == 3, 'class'] = 0  # car
        # self.samples.loc[types == 4, 'class'] = 1  # motorcycle
        # self.samples.loc[types == 6, 'class'] = 2  # bus
        # self.samples.loc[types == 8, 'class'] = 3  # truck
        # self.samples.loc[types == 1, 'class'] = 4  # person
        # self.samples.loc[types == 10, 'class'] = 5  # traffic light
        # self.samples.loc[types == 12, 'class'] = 6  # street sign
        # self.samples.loc[types == 13, 'class'] = 7  # stop sign
        # self.samples.loc[types == 14, 'class'] = 8  # parking meter
        # self.samples.loc[types == 2, 'class'] = 9  # bicyle
        self.samples['class'] = self.samples['class'].astype(int)
        self.samples['frame id'] = self.samples['frame id'].astype(int)
        self.update_idxs()

    def update_idxs(self):
        self.samples["idx"] = pd.Series(
            range(0, len(self.samples["idx"]))).values
        self.samples.set_index("idx", inplace=True, drop=False)

    def __getitem__(self, idx):
        """Get a sample.
        Args
            index (int): Index
        Returns
            tuple: (image, target) where target is a tuple of all target types
                if target_type is a list with more than one item.
        """
        image = self.get_image(idx)
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sample = self.samples.iloc[idx, :]
        if self.label_type == 'human':
            target = sample['class']
        elif self.label_type == 'golden_label':
            target = sample['golden_label']
        else:
            raise RuntimeError(f"Unrecognized label_type: {self.label_type}")
        if self.target_transform:
            image = self.target_transform(target)

        return image, target

    def get_image(self, index):
        '''Read a frame from disk and crop out an object for classification.

        Args
            index(int)

        Return cropped image
        '''
        sample = self.samples.iloc[index, :]
        frame_idx = int(sample['frame id'])
        img_path = os.path.join(self.root, sample['camera'], 'frame_images',
                                "{:06d}.jpg".format(frame_idx))

        cache_file_path = os.path.join(
            self.root, sample['camera'], 'classification_images',
            '{:06d}_{}_{}_{}_{}.jpg'.format(
                int(sample['frame id']), int(sample['xmin']),
                int(sample['ymin']), int(sample['xmax']), int(sample['ymax'])))
        if self.use_cache and os.path.exists(cache_file_path):
            # Cache hit
            cropped = Image.open(cache_file_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
            cropped = image.crop((sample["xmin"], sample["ymin"],
                                  sample["xmax"], sample["ymax"])).resize(
                (self.resize_res, self.resize_res), resample=PIL.Image.BICUBIC)
            cropped.save(cache_file_path)
        if self.transform:
            cropped = self.transform(cropped)
        return cropped

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    def get_md5(self):
        return md5(','.join(
            (str(self.samples['frame id'].values))).encode('utf-8')).hexdigest()

    @property
    def y(self):
        return self.samples["class"].values

    def get_filtered_dataset(self, data_idxs, label_type='human',
                             custom_transforms=None):
        """Subsample the dataset.

        Args
            data_idxs:
            custom_transforms:
        Returns:
            Mp4VideoClassification Object

        """
        trsf = custom_transforms if custom_transforms is not None else \
            self.transform
        dataset = Mp4VideoClassification(
            self.root, self.sample_list_name,
            sample_list_root=self.sample_list_root, subsample_idxs=data_idxs,
            transform=trsf, target_transform=self.target_transform,
            resize_res=self.resize_res, use_cache=self.use_cache,
            label_type=label_type)
        for column_name in self.samples.columns:
            if column_name not in dataset.samples.columns:
                mask = self.samples["idx"].isin(data_idxs)
                col2add = self.samples[mask][column_name]
                dataset.samples[column_name] = col2add
        return dataset

    def get_merged_dataset(self, other_dataset):
        '''Merge this instance of a dataset with another dataset.
        Args
            other_dataset(Mp4VideoClassification)
        Returns
            Mp4VideoClassification object
        '''
        if other_dataset is None:
            return self
        assert isinstance(other_dataset, Mp4VideoClassification), \
            "The other dataset is not long video dataset."
        union_idxs = np.union1d(self.samples["idx"].values,
                                other_dataset.samples["idx"].values)
        return Mp4VideoClassification(
            self.root, self.sample_list_name, subsample_idxs=union_idxs,
            sample_list_root=self.sample_list_root, transform=self.transform,
            target_transform=self.target_transform, resize_res=self.resize_res,
            segment_indices=self.segment_indices, label_type=self.label_type)

    def concat_dataset(self, other_dataset):
        '''
        Concatenates the samples of this dataset with another dataset inplace.
        :param other_dataset:
        :return:
        '''
        if other_dataset is None:
            return self
        assert isinstance(other_dataset, Mp4VideoClassification), "The other dataset is not MP4 dataset."
        self.samples = pd.concat([self.samples, other_dataset.samples], axis=0)
        self.update_idxs()

    def get_time_of_day(self):
        """Return the majority of weather of all samples."""
        return ""

    def get_weather(self):
        """Return the majority of weather of all samples."""
        return ""

    def get_class_dist(self):
        """Return class distribution of all samples."""
        class_cnt = 6
        class_dist = []
        for i in range(class_cnt):
            mask = self.samples['class'] == i
            class_dist.append(len(self.samples[mask]))
        return class_dist

    @staticmethod
    def extract_frame_images(video_file, save_dir):
        if not os.path.exists(save_dir):
            print(f"{save_dir} does not exist. Create {save_dir}.")
            os.mkdir(save_dir)
        img_name = os.path.join(save_dir, "%06d.jpg")
        cmd = "ffmpeg -i {} -qscale:v 1.0 {} -hide_banner".format(
            video_file, img_name)
        subprocess.run(cmd.split(" "), check=True)

    @staticmethod
    def generate_object_detection(video_path, img_dir, output_path,
                                  model_path, device, target_fps=None):
        """Generate sample list.

        Args
            video_path: absolute path a mp4 file.
            img_dir: a directory contain all extracted images.
            output_path: a directory where sample list will be saved to.
            model_path: path to a folder containing tensorflow object detection
                        model.
            device: gpu id.
            target_fps: frame rate used to prepare sample lists.
        """
        import cv2
        from ekya.models.object_detector import ObjectDetector
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        vid = cv2.VideoCapture(video_path)

        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        # load an object detector
        detector = ObjectDetector(model_path, device)
        # load images
        img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        output_name = '{}_{}fps_detections.csv'.format(
            vid_name, fps if target_fps is None else int(round(target_fps)))
        if target_fps is None:
            output_name = '{}_detections.csv'.format(vid_name)
            target_fps = fps
        else:
            output_name = '{}_{}fps_detections.csv'.format(
                vid_name, int(round(target_fps)))
        cnt = 0
        sample_ratio = target_fps / fps

        with open(os.path.join(output_path, output_name), 'w', 1) as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'frame id', 'class', 'xmin', 'ymin',
                             'xmax', 'ymax', 'score', 'timestamp'])
            for i, img_path in enumerate(img_paths):
                frame_id = int(os.path.basename(os.path.splitext(img_path)[0]))
                print('frame {}/{}'.format(frame_id, len(img_paths)))
                if (frame_id - 1) % sample_ratio != 0:
                    continue
                image = np.array(Image.open(img_path))
                dets, t_used = detector.infer(image)
                if dets['num_detections'] != 0:
                    for label, box, score in zip(dets['detection_classes'],
                                                 dets['detection_boxes'],
                                                 dets['detection_scores']):
                        row = [cnt, frame_id, label, box[0], box[1], box[2],
                               box[3], score, (frame_id-1)/fps]
                        writer.writerow(row)
                        cnt += 1

    @staticmethod
    def generate_sample_list_from_detection_file(
            video_path, detection_file, output_path, target_fps=None,
            min_res=20, start_frame=None, end_frame=None):
        """Generate sample list from object detection results.

        Args
            video_path(str): absolute path a mp4 file.
            detection_file(str): file generated by object detetor.
            output_path(str): a directory where sample list will be saved to.
            target_fps(int): frame rate used to prepare sample lists.
            min_res(int): the minimum pixel count of the box height and width.
            start_frame(int): start frame id in the sample list.(1 started)
            end_frame(int): end frame id in the sample list. (inclusive)
        """
        import cv2
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        # load detections from detection file
        if target_fps is None:
            target_fps = fps
        dets = pd.read_csv(detection_file)
        if start_frame is None and end_frame is None:
            output_name = '{}_{}fps.csv'.format(
                vid_name,
                fps if target_fps is None else target_fps)
        else:
            output_name = '{}_{}fps_{}_{}.csv'.format(
                vid_name,
                fps if target_fps is None else target_fps,
                start_frame, end_frame)
        start_frame = 1 if start_frame is None else start_frame
        end_frame = max(dets['frame id'].values) if end_frame is None else end_frame
        # load images

        cnt = 0

        sample_ratio = fps / target_fps

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, output_name), 'w', 1) as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'frame id', 'class', 'xmin', 'ymin',
                             'xmax', 'ymax', 'score', 'timestamp', 'camera'])
            for idx, row in dets.iterrows():
                frame_id = int(row['frame id'])
                if frame_id < start_frame or frame_id > end_frame:
                    continue
                if (frame_id - 1) % sample_ratio != 0:
                    continue
                if ((row['xmax'] - row['xmin'] > min_res) and
                        (row['ymax'] - row['ymin'] > min_res)):
                    row = [cnt, frame_id, row['class'], row['xmin'],
                           row['ymin'], row['xmax'], row['ymax'], row['score'],
                           (frame_id - 1) / fps, vid_name]
                    writer.writerow(row)
                    cnt += 1
