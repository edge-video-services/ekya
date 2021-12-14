"""Aim to test the model cache logic."""
import argparse
import csv
import os
import random

import numpy as np
import torch
from torchvision import transforms

# from ekya.classes.model import RayMLModel
# from ekya.datasets.CityscapesClassification import CityscapesClassification
# from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification
from ekya.datasets.WaymoClassification import WaymoClassification
from ekya.models.resnet import Resnet
from ekya.utils.helpers import read_json_file, seed_all, write_json_file


def get_waymo_dataloader(dataset, num_tasks, task_id, train_batch_size,
                         test_batch_size, num_workers, subsample_rate, shuffle,
                         train_split=0.7):
    """Prepare dataloaders for Waymo Classification Dataset."""
    dataset_idxs = dataset.samples["idx"]
    segments = dataset.samples['segment'].unique().tolist()

    # print("New sample list size: {}. Tasks: {}.".format(
    #     len(dataset_idxs), num_tasks))

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
    # TODO:
    # task_dataset_test = self.dataset.get_filtered_dataset(task_idxs_test)
    # test_loaders[task_id] = torch.utils.data.DataLoader(
    #     task_dataset_test, batch_size=test_batch_size, shuffle=shuffle,
    #     num_workers=num_workers)

    # print('task', task_id, segments)
    # try:
    task = segments[task_id-1]
    # except KeyError:
    #     print(task_id, len(segments))
    #     import pdb
    #     pdb.set_trace()

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
        former_idxs, latter_idxs = dataset.get_split_indices(
            tmp_task, split_time=10)
        if len(former_idxs) > 500:
            former_idxs = np.random.choice(former_idxs, 500).tolist()
        task_data_idxs += former_idxs

    # only current 1st 10s
    former_idxs, latter_idxs = dataset.get_split_indices(
        task, split_time=10)
    if len(former_idxs) > 500:
        former_idxs = np.random.choice(former_idxs, 500)
    else:
        former_idxs = np.array(former_idxs)
    # print('1st 10s has {}, 2nd 10s has {}'.format(
    #     len(former_idxs), len(latter_idxs)))

    # Shuffle and Subsample dataset
    # do not random shuffle
    random.shuffle(task_data_idxs)
    # print("Subsample: {}".format(subsample_rate))
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

    # TODO: use golden model to produce labels here
    task_data_idxs_train = task_data_idxs[:int(
        train_split * len(task_data_idxs))]
    task_data_idxs_val = task_data_idxs[int(
        train_split * len(task_data_idxs)):]
    if dataset.label_type == 'golden_label':
        label_type = 'golden_label'
    else:
        label_type = 'human'

    task_train_dataset = dataset.get_filtered_dataset(
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

    task_val_dataset = dataset.get_filtered_dataset(
        task_data_idxs_val, label_type=label_type)

    # task_val_dataset = ConcatDataset(
    #     [task_val_dataset, pretrained_subsampled_dataset])

    # change batch_size if dataset size is smaller than desired batch_size
    if len(task_val_dataset) < train_batch_size:
        task_dataset_val_loader = torch.utils.data.DataLoader(
            task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers)
    else:
        task_dataset_val_loader = torch.utils.data.DataLoader(
            task_val_dataset, batch_size=train_batch_size, shuffle=shuffle,
            num_workers=num_workers)

    # print(f"Subsampling done. "
    #       f"Task {task} train data: {len(task_train_dataset)}, "
    #       f"val data: {len(task_val_dataset)}, pretrain_data: {0}")
    # len(pretrained_subsampled_dataset)))

    if len(latter_idxs) == 0:
        import pdb
        pdb.set_trace()
    task_idxs_test = latter_idxs
    task_dataset_test = dataset.get_filtered_dataset(
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test model cache experiement script.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['waymo'], help='Dataset name.')
    parser.add_argument("--root", type=str, required=True, help='Dataset root')
    parser.add_argument("--camera_name", type=str, required=True,
                        help="A list of camera/city to be used.")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help='Number of tasks/retrainining windows.')
    parser.add_argument("--hyperparameter", type=int, default=5,
                        help='Hyperparameter id.')
    parser.add_argument("--checkpoint_path", type=str, required=True,)
    parser.add_argument("--num_workers", type=int, default=8,
                        help='Number of workers.')
    parser.add_argument("--hyp_map_path", type=str, required=True,
                        help='path to hyperparameter map.')
    parser.add_argument("--save_path", type=str, required=True,
                        help='path to results.')

    return parser.parse_args()


def get_num_classes(dataset_name):
    if dataset_name == 'cityscapes':
        return 6
    elif dataset_name == 'waymo':
        return 4
    elif dataset_name == 'mp4':
        raise NotImplementedError
    else:
        raise NotImplementedError


def main():
    seed_all(42)
    args = parse_args()
    hyp_map = read_json_file(args.hyp_map_path)
    dataset_name = args.dataset
    num_tasks = args.num_tasks
    hyp_id = str(args.hyperparameter)
    num_classes = get_num_classes(dataset_name)
    chkpt_path = args.checkpoint_path
    root = args.root  # Dataset root
    sample_list_root = os.path.join(root, "sample_lists", "citywise")
    # sample_list_root = os.path.join(root, "sample_lists", "citywise_sorted")
    camera_name = args.camera_name

    # load the dataset
    trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    # if dataset_name == 'cityscapes':
    #     dataset = CityscapesClassification(
    #         root, camera_name, sample_list_root, transform=trsf,
    #         resize_res=224, use_cache=True, label_type='golden_label')
    if dataset_name == 'waymo':
        dataset = WaymoClassification(
            root, camera_name, sample_list_root, transform=trsf,
            resize_res=224, use_cache=True, label_type='golden_label')
    else:
        raise NotImplementedError

    # with open(os.path.join("results_golden_label", f'{camera_name}.csv'), 'w', 1) as f:
    os.makedirs(os.path.join(args.save_path, camera_name), exist_ok=True)
    with open(os.path.join(args.save_path, f'{camera_name}.csv'), 'w', 1) as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow(['task_id', 'cached_camera_name', 'hyp_id', 'cached_camera_task_id', 'test_acc'])
        for hyp_id in ['0', '1', '2', '3', '4', '5']:
            tot_metadata= {}
            for task_id in range(1, num_tasks):
                num_hidden = hyp_map[hyp_id]["num_hidden"]
                last_layer_only = hyp_map[hyp_id]["last_layer_only"]
                model_name = hyp_map[hyp_id]["model_name"]
                batch_size = hyp_map[hyp_id]["batch_size"]
                subsample = hyp_map[hyp_id]["subsample"]

                hyperparams = {
                    'num_hidden': num_hidden,
                    'last_layer_only': last_layer_only,
                    'model_name': model_name
                }
                dataloaders_dict = get_waymo_dataloader(
                    dataset, num_tasks, task_id, batch_size, batch_size, 8, subsample,
                    False)
                metadata = {}
                for k, dataloader in dataloaders_dict.items():
                    metadata[k] = {}
                    metadata[k]['class_distribution'] = dataloader.dataset.get_class_dist()
                    metadata[k]['time_of_day'] = dataloader.dataset.get_time_of_day()
                    metadata[k]['weather'] = dataloader.dataset.get_weather()
                    if k == 'train' and hyp_id == '5':
                        dataloader.dataset.get_image_paths()
                tot_metadata[task_id] = metadata
                test_loader = dataloaders_dict['test']
                for city in ['sf_000_009', 'sf_020_029', 'sf_030_039', 'sf_050_059', 'sf_060_069', 'sf_070_079', 'sf_080_089']:
                    for test_task_id in range(1, 10):
                        selected_model_path = os.path.join(
                            chkpt_path, city,
                            f"config{hyp_id}_task{test_task_id}.pth")
                        model = Resnet(num_classes, hyperparameters=hyperparams,
                                       restore_path=selected_model_path)
                        test_acc = model.infer(test_loader)
                        csv_writer.writerow([task_id, city, hyp_id, test_task_id, test_acc])
            write_json_file(os.path.join(args.save_path, camera_name, f"{hyp_id}_metadata.json"), tot_metadata)


if __name__ == '__main__':
    main()
