import argparse
import csv
import glob
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
from ekya.utils.helpers import read_json_file, seed_all


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
    parser.add_argument("--hyperparameter", type=int, default=5,
                        help='Hyperparameter id.')
    parser.add_argument("--epochs", type=int, default=30,
                        help='Training Epoch.')
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

    test_accs = []
    test_task_ids = []
    models_selected = []
    for task_id in range(num_tasks):
        test_idxs = dataset_idxs[
            num_samples_per_task*task_id:num_samples_per_task*(task_id+1)].values.copy()
        test_set = dataset.get_filtered_dataset(test_idxs, label_type='golden_label')

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, num_workers=args.num_workers)

        class_cnts = []
        for i in range(num_classes):
            mask = test_set.samples['class'] == i
            class_cnts.append(len(test_set.samples[mask]))

        class_dist = np.array(class_cnts) / np.sum(class_cnts)
        classdist_files = []
        similarities = []

        for classdist_file in glob.glob(
                os.path.join(save_path, f"{dataset_name}_aachen_*_class_cnt.csv")):
            if camera_name in classdist_file:
                continue
            classdist_files.append(classdist_file)
            df = pd.read_csv(classdist_file)
            target_class_cnts = df.to_numpy()[0]
            target_class_dist = target_class_cnts / np.sum(target_class_cnts)
            dist = np.linalg.norm(target_class_dist-class_dist)
            similarities.append(dist)
        sorted_idxs = np.argsort(similarities)
        selected_classdist_file = classdist_files[sorted_idxs[0]]
        print(selected_classdist_file)
        _, city, win, _, _ = os.path.splitext(os.path.basename(selected_classdist_file))[0].split('_')
        selected_model_path = os.path.join(
            save_path, f"{dataset_name}_{city}_resnet18_config5_{win}.pth")
        model = Resnet(num_classes, hyperparameters=hyperparams,
                       restore_path=selected_model_path)
        test_acc = model.infer(test_loader)
        test_accs.append(test_acc)
        # if task_id != 0:
        #     retrain_test_acc = retrain_model.infer(test_loader)
        #     retrain_test_accs.append(retrain_test_acc)
        # else:
        #     retrain_test_accs.append(no_retrain_test_acc)
        test_task_ids.append(task_id)
        models_selected.append(selected_model_path)
        # print(f'{camera_name}')
        # print(f'{task_id},{no_retrain_test_acc},{no_retrain_test_acc}')

    df = pd.DataFrame(list(zip(test_task_ids, test_accs, models_selected)),
                      columns=['task id', 'test acc', 'model'])
    df.to_csv(
        os.path.join(save_path, 'test_accs', f'{dataset_name}_{camera_name}_test_accs.csv'), index=False)


if __name__ == '__main__':
    main()
