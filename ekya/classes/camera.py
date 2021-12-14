import os
import random
from typing import List

import ray
import torch
import numpy as np
import pandas as pd
from ray.exceptions import RayActorError

from ekya.CONFIG import RANDOM_SEED, TRAINING_MEMORY_FOOTPRINT, INFERENCE_MEMORY_FOOTPRINT
from ekya.classes.model import RayMLModel
from ekya.utils.dataset_utils import get_dataset
from ekya.datasets.CityscapesClassification import CityscapesClassification
from ekya.datasets.WaymoClassification import WaymoClassification
from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification
from ekya.models.golden_model import GoldenModel

# Set random seed for reproducibility
from ekya.utils.helpers import seed_all, seed_python

seed_all(RANDOM_SEED)


class Camera(object):
    def __init__(self, id: str,
                 train_sample_names: List[str],
                 sample_list_path: str,
                 num_tasks: int,
                 train_split: float,
                 pretrained_sample_names: List[str] = [],
                 dataset_name: str = "cityscapes",
                 dataset_root: str = "",
                 inference_profile_path: str = "",
                 max_inference_resources: float = 0.25,
                 label_type: str = "human",
                 golden_model_ckpt: str = ""):
        '''
        :param id: String id of the camera.
        :param train_sample_names: Sample lists to use as input.
        :param sample_list_path: Path to the samplelist root
        :param num_tasks: Number of tasks in the camera
        :param train_split: Split between train:val
        :param pretrained_sample_names: Samples lists used for pre-training. Used in retraining.
        :param dataset_name: Either of cityscapes or waymo
        :param dataset_root: Root of the dataset
        :param inference_profile_path: Path of the inference profiles.
        :param max_inference_resources: Ceiling of inference resources required. When fewer than this resources are allocated, the inference accuracy will be scaled by the inference profile function.
        '''
        self.id = id
        self.train_sample_names = train_sample_names
        dataset_class, dataset_default_args = get_dataset(dataset_name)
        self.dataset_root = dataset_root or dataset_default_args["root"]
        self.num_classes = dataset_default_args["num_classes"]
        self.sample_list_path = sample_list_path or dataset_default_args['sample_list_root']
        self.golden_model_ckpt = golden_model_ckpt
        print(self.dataset_root)
        self.dataset = dataset_class(root=self.dataset_root,
                                     sample_list_name=train_sample_names,
                                     sample_list_root=self.sample_list_path,
                                     transform=dataset_default_args['trsf'],
                                     resize_res=224,
                                     use_cache=dataset_default_args['use_cache'], label_type=label_type)
        print("[Camera {}] Using train samples from {}\nDataset size {}".format(self.id,
                                                                                train_sample_names,
                                                                                len(self.dataset)))
        self.pretrained_sample_names = pretrained_sample_names
        if pretrained_sample_names:
            if label_type == 'golden_label' and self.golden_model_ckpt != "":
                self.dataset_pretrained = dataset_class(
                    root=self.dataset_root,
                    sample_list_name=pretrained_sample_names,
                    sample_list_root=self.sample_list_path,
                    transform=dataset_default_args['trsf'], resize_res=224,
                    use_cache=dataset_default_args['use_cache'])
                loader = torch.utils.data.DataLoader(
                    self.dataset_pretrained, batch_size=64, num_workers=8)
                golden_labels = generate_golden_model_labels(
                    loader, self.golden_model_ckpt)
                self.dataset_pretrained.samples.loc[:, 'golden_label'] = golden_labels
                self.dataset_pretrained.label_type = label_type
            else:
                self.dataset_pretrained = dataset_class(
                    root=self.dataset_root,
                    sample_list_name=pretrained_sample_names,
                    sample_list_root=self.sample_list_path,
                    transform=dataset_default_args['trsf'], resize_res=224,
                    use_cache=dataset_default_args['use_cache'],
                    label_type=label_type)
            print("[Camera {}] Using pretrained samples for retraining: {}\nPretrained dataset size {}".format(self.id,
                                                                                                            pretrained_sample_names,
                                                                                                            len(self.dataset_pretrained)))
        self.num_tasks = num_tasks
        self.current_task = -1  # -1 = not started.
        self.train_split = train_split
        self.setup_dataset()
        self.current_inference_path = ""
        self.dataset_name = dataset_name

        # Read inference profile if any. Else use no profiles
        self.inference_profile_path = inference_profile_path
        self.max_inference_resources = max_inference_resources
        if self.inference_profile_path:
            self.inference_scaling_function = self.get_infer_profile(self.max_inference_resources,
                                   self.inference_profile_path,
                                   camera='c1')
            print("[Camera] Using a scaling function with max {}".format(self.max_inference_resources))
        else:
            print("[Camera] No scaling function provided. Using no scaling.")
            self.inference_scaling_function = lambda x: 1   # No scaling if inference profile is not provided.

    @staticmethod
    def get_infer_profile(max_inference_resources=1,
                          profile_path='real_inference_profiles.csv',
                          camera='c1'):
        '''
        :param max_inference_resources:
        :param profile_path:
        :param camera:
        :return: The return function's range should be [0,1]
        '''
        data = pd.read_csv(profile_path)
        # effective_subsample_rate is max_inference_resources/res + 0.0000001 to avoid divide by zero.
        return lambda res: np.interp(max_inference_resources / (res + 0.000001), data['subsampling'], data[camera],
                                     right=0)

    def setup_dataset(self):
        self.dataset_idxs = self.dataset.samples["idx"]
        self.num_samples_per_task = int(len(self.dataset_idxs) / self.num_tasks)

    def _get_cityscapes_dataloader(self, task_id, train_batch_size,
                                   test_batch_size, num_workers,
                                   subsample_rate, shuffle):
        if task_id == 0:
            # If first task, no retraining and validation just test
            task_dataset_train_loader = None
            task_dataset_val_loader = None
        else:
            # task_data_idxs = self.dataset_idxs[self.num_samples_per_task * (task_id - 1):self.num_samples_per_task * task_id].values.copy()
            # We do not include data history. The following line implements
            # that, uncomment it and comment the above one if you want to use
            # entire history.
            task_data_idxs = self.dataset_idxs[
                0:self.num_samples_per_task * task_id].values.copy()
            # print(task_data_idxs)
            # random.shuffle(task_data_idxs)

            # Pick at least two samples.
            num_samples_to_pick = max(int(
                len(task_data_idxs) * subsample_rate), 2)
            task_data_subsampled_idxs = np.random.choice(
                task_data_idxs, num_samples_to_pick, replace=False)

            # Make sure to leave atleast one sample for validation
            train_end_idx = min(
                int(self.train_split * len(task_data_subsampled_idxs)),
                len(task_data_subsampled_idxs) - 1)
            task_data_idxs_train = task_data_subsampled_idxs[:train_end_idx]
            # Validation: all samples except training samples
            task_data_idxs_val = [x for x in task_data_idxs if x not in task_data_idxs_train]

            # task_data_idxs_train = task_data_idxs[:int(self.train_split * len(task_data_idxs))]
            # task_data_idxs_val = task_data_idxs[int(self.train_split * len(task_data_idxs)):]

            if self.dataset.label_type == 'golden_label':
                label_type = 'golden_label'
            else:
                label_type = 'human'
            # use golden model label to retrain
            task_train_dataset = self.dataset.get_filtered_dataset(
                task_data_idxs_train, label_type=label_type)
            task_val_dataset = self.dataset.get_filtered_dataset(
                task_data_idxs_val, label_type=label_type)

            if self.pretrained_sample_names:
                # Setup pretrained dataset
                pretrained_subsample_idxs = self.dataset_pretrained.samples["idx"].values

                # Pick at least two samples.
                num_samples_to_pick = max(
                    int(len(pretrained_subsample_idxs) * subsample_rate), 2)
                pretrained_subsample_idxs = np.random.choice(
                    pretrained_subsample_idxs, num_samples_to_pick,
                    replace=False)
                pretrained_subsampled_dataset = self.dataset_pretrained.get_filtered_dataset(
                    pretrained_subsample_idxs, label_type=label_type)

                # Concat to train and validation
                task_train_dataset.concat_dataset(pretrained_subsampled_dataset)
                task_val_dataset.concat_dataset(pretrained_subsampled_dataset)

            task_dataset_train_loader = torch.utils.data.DataLoader(
                task_train_dataset, batch_size=train_batch_size,
                shuffle=shuffle, num_workers=num_workers)
            task_dataset_val_loader = torch.utils.data.DataLoader(
                task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
                num_workers=num_workers)

        task_idxs_test = self.dataset_idxs[
                         self.num_samples_per_task * task_id:
                         self.num_samples_per_task * (task_id + 1)]
        if self.dataset.label_type == 'golden_label':
            label_type = 'golden_label'
        else:
            label_type = 'human'
        # use golden model to test
        task_dataset_test = self.dataset.get_filtered_dataset(
            task_idxs_test, label_type=label_type)
        task_dataset_test_loader = torch.utils.data.DataLoader(
            task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
            num_workers=num_workers)
        dataloaders_dict = {'train': task_dataset_train_loader,
                            'val': task_dataset_val_loader,
                            'test': task_dataset_test_loader}
        print("Task {}. Train data: {}, val data: {}, test data: {}, "
              "Subsample {}.".format(
                task_id, len(task_train_dataset), len(task_val_dataset),
                len(task_dataset_test), subsample_rate))
        return dataloaders_dict

    def _get_waymo_dataloader(self, task_id, train_batch_size, test_batch_size,
                              num_workers, subsample_rate, shuffle):
        """Prepare dataloaders for Waymo Classification Dataset."""
        dataset_idxs = self.dataset.samples["idx"]
        segments = self.dataset.samples['segment'].unique().tolist()

        print("New sample list size: {}. Tasks: {}.".format(
            len(dataset_idxs), self.num_tasks))

        # Since we need to test on all tasks, generate a map of taskid to
        # loaders before execution:
        # assert len(segments) >= self.num_tasks, \
        #     f"Number of Waymo segments = {len(segments)}." \
        #     f"Ekya Task number = {self.num_tasks}. Number of Waymo Segment " \
        #     "must greater than the number of Ekya tasks!"

        # test_loaders = {}   # Map of taskid to it's test loader
        # task = segments[task_id]
        # print(task)
        # mask = self.dataset.samples['segment'] == task
        # task_idxs_test = self.dataset.samples[mask]['idx']
        # Deprecated:
        # former_idxs, latter_idxs = dataset.get_split_indices(
        #     task_idxs_test, split_time=10)
        # task_dataset_test = dataset.get_filtered_dataset(latter_idxs)

        # test
        # task_dataset_test = self.dataset.get_filtered_dataset(task_idxs_test)
        # test_loaders[task_id] = torch.utils.data.DataLoader(
        #     task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
        #     num_workers=num_workers, drop_last=True)

        print('task', task_id, segments)
        try:
            task = segments[task_id-1]
        except KeyError:
            print(task_id, len(segments))

        # Get train set from previous task
        # if task_idx == task_offset:
        task_dataset_train_loader = None
        task_dataset_val_loader = None
        # We include data history too to avoid catastrophic forgetting?
        # task_data_idxs = dataset_idxs[num_samples_per_task *
        # (task - 1):num_samples_per_task * task].values
        # accumulate the previous task's data
        task_data_idxs = []
        for tmp_task in segments[0:task_id-1]:
            former_idxs, latter_idxs = self.dataset.get_split_indices(
                tmp_task, split_time=10)
            if len(former_idxs) > 500:
                former_idxs = np.random.choice(former_idxs, 500).tolist()
            task_data_idxs += former_idxs

        # only current 1st 10s
        former_idxs, latter_idxs = self.dataset.get_split_indices(
            task, split_time=10)
        if len(former_idxs) > 500:
            former_idxs = np.random.choice(former_idxs, 500)
        else:
            former_idxs = np.array(former_idxs)
        print('1st 10s has {}, 2nd 10s has {}'.format(
            len(former_idxs), len(latter_idxs)))

        # Shuffle and Subsample dataset
        # do not random shuffle
        random.shuffle(task_data_idxs)
        print("Subsample: {}".format(subsample_rate))
        task_data_idxs = np.random.choice(
            task_data_idxs, int(len(task_data_idxs)*subsample_rate),
            replace=False)
        task_data_idxs = np.concatenate([task_data_idxs, former_idxs])
        # task_data_idxs = sorted(task_data_idxs)

        # pretrained_subsample_idxs = \
        #     dataset_for_pretrained.samples["idx"].values
        # pretrained_subsample_idxs = np.random.choice(
        #     pretrained_subsample_idxs,
        #     int(len(pretrained_subsample_idxs)*subsample),
        #     replace=False)
        # pretrained_subsampled_dataset = \
        #     dataset_for_pretrained.get_filtered_dataset(
        #         pretrained_subsample_idxs)

        task_data_idxs_train = task_data_idxs[:int(
            self.train_split * len(task_data_idxs))]
        task_data_idxs_val = task_data_idxs[int(
            self.train_split * len(task_data_idxs)):]
        if self.dataset.label_type == 'golden_label':
            label_type = 'golden_label'
        else:
            label_type = 'human'

        task_train_dataset = self.dataset.get_filtered_dataset(
            task_data_idxs_train, label_type=label_type)
        # task_train_dataset = ConcatDataset(
        #     [task_train_dataset, pretrained_subsampled_dataset])
        if len(task_train_dataset) < train_batch_size:
            task_dataset_train_loader = torch.utils.data.DataLoader(
                task_train_dataset, batch_size=train_batch_size,
                shuffle=shuffle, num_workers=num_workers)
        else:
            task_dataset_train_loader = torch.utils.data.DataLoader(
                task_train_dataset, batch_size=train_batch_size,
                shuffle=shuffle, num_workers=num_workers)

        task_val_dataset = self.dataset.get_filtered_dataset(
            task_data_idxs_val, label_type=label_type)
        # task_val_dataset = ConcatDataset(
        #     [task_val_dataset, pretrained_subsampled_dataset])
        if len(task_val_dataset) < train_batch_size:
            task_dataset_val_loader = torch.utils.data.DataLoader(
                task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
                num_workers=num_workers)
        else:
            task_dataset_val_loader = torch.utils.data.DataLoader(
                task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
                num_workers=num_workers)

        print(f"Subsampling done. "
              f"Task {task} train data: {len(task_train_dataset)}, "
              f"val data: {len(task_val_dataset)}, pretrain_data: {0}")
        # len(pretrained_subsampled_dataset)))

        if len(latter_idxs) == 0:
            import pdb
            pdb.set_trace()
        task_idxs_test = latter_idxs
        task_dataset_test = self.dataset.get_filtered_dataset(
            task_idxs_test, label_type=label_type)
        if len(task_dataset_test) < test_batch_size:
            task_dataset_test_loader = torch.utils.data.DataLoader(
                task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
                num_workers=num_workers)
        else:
            task_dataset_test_loader = torch.utils.data.DataLoader(
                task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
                num_workers=num_workers)

        # NOTE: Adding test every epoch because profiling
        dataloaders_dict = {'train': task_dataset_train_loader,
                            'val': task_dataset_val_loader,
                            'test': task_dataset_test_loader}

        return dataloaders_dict

    def _get_mp4_dataloader(self, task_id, train_batch_size, test_batch_size,
                            num_workers, subsample_rate, shuffle):
        if task_id == 0:
            # If first task, no retraining and validation just test
            task_dataset_train_loader = None
            task_dataset_val_loader = None
        else:
            # We do not include data history. The following line implements
            # that, uncomment it and comment the above one if you want to use
            # entire history.
            task_data_idxs = self.dataset_idxs[
                0:self.num_samples_per_task * task_id].values.copy()
            if len(task_data_idxs) > 2000:
                task_data_idxs = np.random.choice(
                    task_data_idxs, 2000, replace=False)
            # task_data_idxs = self.dataset_idxs[
            #     self.num_samples_per_task * (task_id - 1):self.num_samples_per_task * task_id].values.copy()
            # TODO: Uncomment shuffle
            # random.shuffle(task_data_idxs)

            # Pick at least two samples.
            num_samples_to_pick = max(int(
                len(task_data_idxs) * subsample_rate), 2)
            task_data_subsampled_idxs = np.random.choice(
                task_data_idxs, num_samples_to_pick, replace=False)
            # task_data_subsampled_idxs = np.sort(task_data_subsampled_idxs)

            # Make sure to leave atleast one sample for validation
            train_end_idx = min(
                int(self.train_split * len(task_data_subsampled_idxs)),
                len(task_data_subsampled_idxs) - 1)
            task_data_idxs_train = task_data_subsampled_idxs[:train_end_idx]
            # Validation: all samples except training samples
            # TODO(romilb): Shouldnt this be from task_data_subsampled_idxs?
            task_data_idxs_val = [
                x for x in task_data_idxs if x not in task_data_idxs_train]

            # task_data_idxs_train = task_data_idxs[:int(self.train_split * len(task_data_idxs))]
            # task_data_idxs_val = task_data_idxs[int(self.train_split * len(task_data_idxs)):]

            if self.dataset.label_type == 'golden_label':
                label_type = 'golden_label'
            else:
                label_type = 'human'
            # use golden model label to retrain
            task_train_dataset = self.dataset.get_filtered_dataset(
                task_data_idxs_train, label_type=label_type)
            task_val_dataset = self.dataset.get_filtered_dataset(
                task_data_idxs_val, label_type=label_type)

            if self.pretrained_sample_names:
                # Setup pretrained dataset
                pretrained_subsample_idxs = self.dataset_pretrained.samples["idx"].values

                # Pick at least two samples.
                num_samples_to_pick = max(
                    int(len(pretrained_subsample_idxs) * subsample_rate), 2)
                pretrained_subsample_idxs = np.random.choice(
                    pretrained_subsample_idxs, num_samples_to_pick,
                    replace=False)
                pretrained_subsampled_dataset = self.dataset_pretrained.get_filtered_dataset(
                    pretrained_subsample_idxs, label_type=label_type)

                # Concat to train and validation
                task_train_dataset.concat_dataset(
                    pretrained_subsampled_dataset)
                task_val_dataset.concat_dataset(pretrained_subsampled_dataset)

            task_dataset_train_loader = torch.utils.data.DataLoader(
                task_train_dataset, batch_size=train_batch_size,
                shuffle=shuffle, num_workers=num_workers)
            task_dataset_val_loader = torch.utils.data.DataLoader(
                task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
                num_workers=num_workers)

        task_idxs_test = self.dataset_idxs[
            self.num_samples_per_task * task_id:
            self.num_samples_per_task * (task_id + 1)]
        if self.dataset.label_type == 'golden_label':
            label_type = 'golden_label'
        else:
            label_type = 'human'
        # use golden model to test
        task_dataset_test = self.dataset.get_filtered_dataset(
            task_idxs_test, label_type=label_type)
        task_dataset_test_loader = torch.utils.data.DataLoader(
            task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
            num_workers=num_workers)

        dataloaders_dict = {'train': task_dataset_train_loader,
                            'val': task_dataset_val_loader,
                            'test': task_dataset_test_loader}

        print("Task {}. Train data: {}, val data: {}, test data: {}, "
              "Subsample {}.".format(
                  task_id, len(task_train_dataset), len(task_val_dataset),
                  len(task_dataset_test), subsample_rate))
        return dataloaders_dict

    def _get_dataloader(self,
                        task_id: int,
                        train_batch_size: int = 1,
                        test_batch_size: int = 1,
                        num_workers: int = 0,
                        subsample_rate: float = 1,
                        shuffle: bool = False) -> dict:
        seed_all(RANDOM_SEED)
        if isinstance(self.dataset, CityscapesClassification):
            return self._get_cityscapes_dataloader(
                task_id, train_batch_size, test_batch_size, num_workers,
                subsample_rate, shuffle)
        elif isinstance(self.dataset, WaymoClassification):
            return self._get_waymo_dataloader(
                task_id, train_batch_size, test_batch_size, num_workers,
                subsample_rate, shuffle)
        elif isinstance(self.dataset, Mp4VideoClassification):
            return self._get_mp4_dataloader(
                task_id, train_batch_size, test_batch_size, num_workers,
                subsample_rate, shuffle)

    def update_training_model(self,
                              hyperparameters: dict,
                              training_gpu_weight: float,
                              ray_resource_demand: float,
                              restore_path: str = "",
                              blocking: bool = False):
        self.hyperparameters = hyperparameters
        if "num_classes" in self.hyperparameters:
            print(f"Warning: Overriding num_classes from {self.hyperparameters} to {self.num_classes}")
        self.hyperparameters['num_classes'] = self.num_classes
        self.training_gpu_weight = training_gpu_weight
        self.training_ray_demand = ray_resource_demand
        # Kill the existing model by waiting it to terminate any running tasks
        if hasattr(self, "training_model"):
            try:
                ray.get(self.training_model.__ray_terminate__.remote())
                ray.kill(self.training_model, no_restart=True)
            except RayActorError:
                # Model already killed
                pass
        model_updated = False
        while not model_updated:
            try:
                self.training_model = RayMLModel.options(name="{}_training".format(self.id), num_gpus=self.training_ray_demand).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.training_gpu_weight,
                    restore_path=restore_path,
                    name="{}_training".format(self.id))
                model_updated = True
            except ValueError as e:
                # print("Got value error {}. Retrying..".format(e))
                pass  # Retrying because of actor name failures
        if blocking:
            print("WARNING: Training model init is blocking=True. This may cause training jobs to start before the clock timer starts.")
            ray.get(self.training_model.ready.remote())

    def update_inference_model(self,
                               hyperparameters: dict,
                               inference_gpu_weight: float,
                               ray_resource_demand: float,
                               restore_path: str = "",
                               blocking: bool = False):
        self.hyperparameters = hyperparameters
        if "num_classes" in self.hyperparameters:
            print(f"Warning: Overriding num_classes from {self.hyperparameters} to {self.num_classes}")
        self.hyperparameters['num_classes'] = self.num_classes
        self.inference_gpu_weight = inference_gpu_weight
        self.inference_ray_demand = ray_resource_demand
        print("{}, Update, setting inference wt: {}. {}".format(self.id, self.inference_gpu_weight, self))
        self.current_inference_path = restore_path or self.current_inference_path  # Use restore path if specified, else use current_inference_path
        # Kill the existing model by waiting it to terminate any running tasks
        if hasattr(self, "inference_model"):
            try:
                ray.get(self.inference_model.__ray_terminate__.remote())
            except RayActorError as e:
                # Model already killed
                print("Got exception while killing model. Continuing. {}".format(str(e)))
                pass
        model_updated = False
        while not model_updated:
            try:
                self.inference_model = RayMLModel.options(name="{}_inference".format(self.id), num_gpus=self.inference_ray_demand).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.inference_gpu_weight,
                    inference_scaling_function=self.inference_scaling_function,
                    restore_path=self.current_inference_path,
                    name="{}_inference".format(self.id))
                model_updated = True
            except ValueError as e:
                pass  # Retrying
        if blocking:
            print("WARNING: Model init is blocking=True. This may cause training jobs to start before the clock timer starts.")
            ray.get(self.inference_model.ready.remote())

    def set_current_task(self, new_current_task: int):
        '''
        Updates the current task for this camera. This method is called when a task is incremented.
        :return:
        '''
        self.current_task = new_current_task
        # run golden model inference if the camera is using goden label
        if self.dataset.label_type == 'golden_label' and self.golden_model_ckpt:
            # prepare the dataloader here
            if isinstance(self.dataset, CityscapesClassification) or isinstance(self.dataset, Mp4VideoClassification):
                if self.current_task == 1:
                    task_data_idxs = self.dataset_idxs[
                        self.num_samples_per_task * (self.current_task - 1):
                        self.num_samples_per_task * (self.current_task + 1)].values.copy()
                else:
                    task_data_idxs = self.dataset_idxs[
                        self.num_samples_per_task * self.current_task:
                        self.num_samples_per_task * (self.current_task + 1)].values.copy()
            elif isinstance(self.dataset, WaymoClassification):
                segments = self.dataset.samples['segment'].unique().tolist()
                mask = self.dataset.samples['segment'] == segments[self.current_task - 1]
                task_data_idxs = self.dataset.samples[mask]["idx"]

            dataset = self.dataset.get_filtered_dataset(task_data_idxs)

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, num_workers=8)
            print("Generating golden labels")
            golden_labels = generate_golden_model_labels(
                loader, self.golden_model_ckpt)
            mask = self.dataset.samples["idx"].isin(task_data_idxs)
            if 'golden_label' not in self.dataset.samples.columns:
                # doing this because missing values will be filled with NaN
                # but NaN can only be treated as float causing the entire
                # column to be float
                self.dataset.samples['golden_label'] = -1
            self.dataset.samples.loc[mask, 'golden_label'] = golden_labels
            golden_accuracy = (self.dataset.samples.loc[mask, 'golden_label'] == self.dataset.samples.loc[mask, 'class']).sum()/len(golden_labels)
            print("Accuracy of golden model for task {}: {}".format(self.current_task, golden_accuracy))

    def run_retraining(self,
                       hyperparameters: dict,
                       training_gpu_weight: float,
                       ray_resource_demand: float,
                       dataloaders_dict: dict = {},
                       validation_freq: int = -1,
                       restore_path: str = "",
                       profiling_mode: bool = False,
                       model_save_dir="") -> ray.ObjectID:
        print("Starting retraining for camera {}".format(self.id))
        self.update_training_model(hyperparameters, training_gpu_weight, ray_resource_demand, restore_path=restore_path, blocking=False)
        if not dataloaders_dict:
            dataloaders_dict = self._get_dataloader(self.current_task,
                                                    train_batch_size=hyperparameters["train_batch_size"],
                                                    subsample_rate=hyperparameters["subsample"])
        if profiling_mode:
            validation_freq = int(hyperparameters["validation_freq"])
        metadata = {}
        for k, dataloader in dataloaders_dict.items():
            metadata[k] = {}
            metadata[k]['class_distribution'] = dataloader.dataset.get_class_dist()
            metadata[k]['time_of_day'] = dataloader.dataset.get_time_of_day()
            metadata[k]['weather'] = dataloader.dataset.get_weather()

        task = self.training_model.retrain_model.remote(
            dataloaders_dict['train'], dataloaders_dict['val'],
            dataloaders_dict['test'], hyperparameters,
            validation_freq, profiling_mode)
        # TODO: cache retrained models
        if profiling_mode:
            model_save_path = os.path.join(model_save_dir, "config{}_task{}.pth".format(hyperparameters['id'], self.current_task))
            self.training_model.save_model.remote(model_save_path)
        return task, metadata

    def update_inference_from_retrained_model(self,
                                              path: str = None):
        if not path:
            path = '/tmp/ckpt_{}.pth'.format(self.id)
        ray.get(self.training_model.save_model.remote(path))
        # Kill training model after it is saved.
        try:
            ray.get(self.training_model.__ray_terminate__.remote())
        except RayActorError as e:
            # Model already killed
            print("Got exception while killing model. Continuing. {}".format(str(e)))
            pass
        self.current_inference_path = path
        print("{}, request update {}".format(self.id, self))
        self.update_inference_model(self.hyperparameters,
                                    self.inference_gpu_weight,
                                    self.inference_ray_demand,
                                    restore_path=self.current_inference_path)

    def training_memory_footprint(self):
        return TRAINING_MEMORY_FOOTPRINT


    def inference_memory_footprint(self):
        return INFERENCE_MEMORY_FOOTPRINT


def generate_golden_model_labels(dataloader, checkpoint_path,  # , device,
                                 golden_model_name='resnext101_elastic'):
    """Use golden model to produce retraining labels.
    Args
        dataloader: pytorch dataloader of the training data.
        checkpoint_path: path to golden model weights.
        device: GPU id.
        golden_model_name: name of golden model. Default: resnext101_elasticA.
    """
    if isinstance(dataloader.dataset, WaymoClassification):
        dataset_name = 'waymo'
    elif isinstance(dataloader.dataset, CityscapesClassification):
        dataset_name = 'cityscapes'
    elif isinstance(dataloader.dataset, Mp4VideoClassification):
        dataset_name = 'mp4'
    else:
        raise NotImplementedError("Uknown dataset class.")
    model = GoldenModel(golden_model_name, checkpoint_path, dataset_name)
    _, golden_labels = model.infer(dataloader)
    torch.cuda.empty_cache()
    return golden_labels
