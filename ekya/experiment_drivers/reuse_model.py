import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torchvision import transforms

from ekya.datasets.CityscapesClassification import CityscapesClassification
from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification
from ekya.datasets.WaymoClassification import WaymoClassification
from ekya.models.resnet import Resnet
from ekya.utils.helpers import seed_all, read_json_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Section2 model customization experiement script")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cityscapes', 'waymo', 'mp4'],
                        help='Dataset name.')
    parser.add_argument("--root", type=str, required=True, help='Dataset root')
    parser.add_argument("--camera_name", type=str, required=True,
                        help="A list of camera/city to be used.")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help='Number of tasks/retrainining windows.')
    # parser.add_argument("--num_tasks_to_train", type=int, default=3,
    #                     help='Number of tasks/retrainining windows from start'
    #                     ' to train the model.')
    # parser.add_argument("--batch_size", type=int, default=16,
    #                     help='Batch size.')
    parser.add_argument("--hyperparameter", type=int, default=5,
                        help='Hyperparameter id.')
    parser.add_argument("--epochs", type=int, default=30,
                        help='Training Epoch.')
    # parser.add_argument("--learning_rate", type=float, default=0.001,
    #                     help='Training Learning Rate.')
    # parser.add_argument("--momentum", default=0.9, type=float,
    #                     help="Momentum.")
    #
    # parser.add_argument("--last_layer_only", action='store_true',
    #                     help='Retrain last layer only of if specified.')
    # parser.add_argument("--model_name", type=str, required=True,
    #                     choices=['resnet18', 'resnet50', 'resnet101',
    #                              'resnet152'],
    #                     help='Model name.')
    # parser.add_argument("--num_hidden", type=int, default=64,
    #                     help='Number of hidden neuron in the last layer.')
    parser.add_argument("--checkpoint_path", type=str, required=True,)
    parser.add_argument("--num_workers", type=int, default=8,
                        help='Number of workers.')
    parser.add_argument("--no_train_model_path", type=str, required=True,
                        help='model customized on other cameras')
    parser.add_argument("--hyp_map_path", type=str, required=True,
                        help='path to hyperparameter map.')

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
    # num_tasks_to_train = args.num_tasks_to_train
    hyp_id = str(args.hyperparameter)
    num_classes = get_num_classes(dataset_name)
    num_epochs = args.epochs
    save_path = args.checkpoint_path
    root = args.root  # Dataset root
    sample_list_root = os.path.join(root, "sample_lists", "citywise")
    # sample_list_root = os.path.join(root, "sample_lists", "citywise_sorted")
    camera_name = args.camera_name
    no_train_model_path = args.no_train_model_path
    # val_sample_names = args["lists_val"].split(',')

    num_hidden = hyp_map[hyp_id]["num_hidden"]
    last_layer_only = hyp_map[hyp_id]["last_layer_only"]
    model_name = hyp_map[hyp_id]["model_name"]
    lr = hyp_map[hyp_id]["learning_rate"]
    momentum = hyp_map[hyp_id]["momentum"]
    batch_size = hyp_map[hyp_id]["batch_size"]

    hyperparams = {
        'num_hidden': num_hidden,
        'last_layer_only': last_layer_only,
        'model_name': model_name
    }

    # load the dataset
    trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    if dataset_name == 'cityscapes':
        dataset = CityscapesClassification(
            root, camera_name, sample_list_root, transform=trsf,
            resize_res=224, use_cache=True, label_type='golden_label')
    elif dataset_name == 'waymo':
        dataset = WaymoClassification(
            root, camera_name, sample_list_root, transform=trsf,
            resize_res=224, use_cache=True)
    else:
        raise NotImplementedError
    print(f'loaded {dataset_name} {camera_name}')
    num_samples_per_task = int(round(len(dataset) / num_tasks))
    dataset_idxs = dataset.samples['idx']

    for task_id in range(num_tasks):

        if task_id != 0:
            model = Resnet(num_classes, hyperparameters=hyperparams,
                           restore_path=no_train_model_path)
            train_val_idxs = dataset_idxs[
                max((task_id-2), 0):num_samples_per_task * task_id].values.copy()

            train_val_idxs = np.random.choice(
                train_val_idxs, len(train_val_idxs), replace=False)
            train_idxs = train_val_idxs[:int(0.75 * len(train_val_idxs))]
            val_idxs = train_val_idxs[int(0.75 * len(train_val_idxs)):]

            train_set = dataset.get_filtered_dataset(train_idxs, label_type='golden_label')
            val_set = dataset.get_filtered_dataset(val_idxs, label_type='golden_label')

            classdist_save_path = os.path.join(
                save_path, f"{dataset_name}_{camera_name}_win{task_id}_class_cnt.csv")
            class_cnts = []
            for i in range(num_classes):
                mask = train_set.samples['class'] == i
                class_cnts.append(len(train_set.samples[mask]))
            with open(classdist_save_path, 'w', 1) as f:
                # 'person': 0
                # 'car': 1
                # 'truck': 2
                # 'bus': 3
                # 'bicycle': 4
                # 'motorcycle': 5
                writer = csv.writer(f)
                writer.writerow(['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'])
                writer.writerow(class_cnts)

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers)

            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers)

            dataloaders_dict = {'train': train_loader,
                                'val': val_loader}

            print(f'train {dataset_name} {camera_name} {task_id}')
            _, val_acc_history, best_acc, profile, _ = \
                model.train_model(
                    dataloaders_dict, num_epochs=num_epochs, lr=lr,
                    momentum=momentum)

            model_save_path = os.path.join(
                save_path,
                f'{dataset_name}_{camera_name}_{model_name}_config{hyp_id}_win{task_id}.pth')
            model.save(model_save_path)
            # meta_path = os.path.join(
            #     save_path,
            #     f'{dataset_name}_{camera_name}_{model_name}_{num_hidden}_{num_tasks}'
            #     f'tasks_train{num_tasks_to_train}tasks.json')
            # with open(meta_path, 'w') as f:
            #     json.dump(meta, f)
        # test_idxs = dataset_idxs[
        #     num_samples_per_task*task_id:num_samples_per_task*(task_id+1)].values.copy()
        # test_set = dataset.get_filtered_dataset(test_idxs)
        #
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=batch_size, num_workers=args.num_workers)
        # no_retrain_test_acc = no_retrain_model.infer(test_loader)
        # no_retrain_test_accs.append(no_retrain_test_acc)
        # if task_id != 0:
        #     retrain_test_acc = retrain_model.infer(test_loader)
        #     retrain_test_accs.append(retrain_test_acc)
        # else:
        #     retrain_test_accs.append(no_retrain_test_acc)
        # test_task_ids.append(task_id)
        # print(f'{camera_name}')
        # print(f'{task_id},{no_retrain_test_acc},{no_retrain_test_acc}')


if __name__ == '__main__':
    main()
