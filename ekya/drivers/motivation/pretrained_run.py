import json
import random
from collections import defaultdict

from ekya.models.resnet import Resnet
import torch
from torchvision import transforms
import os
from inclearn.lib.CityscapesClassification import CityscapesClassification
from ekya.drivers.motivation.parser import get_parser

def pretrained_run(args):
    NUM_CLASSES = args["num_classes"]
    BATCH_SIZE = args["batch_size"]
    root = args["root"]  # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    val_sample_names = args["lists_val"].split(',')
    cache = args["use_data_cache"]
    num_tasks = args["num_tasks"]
    results_path = args["results_path"]
    restore_path = args["restore_path"]

    hyperparams = {'num_hidden': args["num_hidden"],
                            'last_layer_only': not args["disable_last_layer_only"],
                            'model_name': args["model_name"]}

    model = Resnet(NUM_CLASSES, restore_path=restore_path, hyperparameters=hyperparams)

    trsf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    dataset = CityscapesClassification(root, val_sample_names, sample_list_root, transform=trsf, resize_res=224,
                                       use_cache=cache)
    dataset_idxs = dataset.samples["idx"]
    num_samples_per_task = int(len(dataset_idxs) / num_tasks)

    results = defaultdict(dict)

    for task in range(num_tasks):
        print("Task {}/{}".format(task, num_tasks))
        # Get train set from previous task
        task_idxs_test = dataset_idxs[num_samples_per_task * task:num_samples_per_task * (task + 1)]
        task_dataset_test = dataset.get_filtered_dataset(task_idxs_test)
        task_dataset_test_loader = torch.utils.data.DataLoader(task_dataset_test, batch_size=BATCH_SIZE, shuffle=True,
                                                               num_workers=args['num_workers'])
        # Testing
        print("Testing")
        test_acc = model.infer(task_dataset_test_loader)

        print("Task {} done.\nTest acc:\t{}".format(task, test_acc))
        results["test_acc"][task] = test_acc

    results = dict(results)
    print(results)
    with open(results_path, 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    pretrained_run(args)