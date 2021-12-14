import json
import random
from collections import defaultdict

from ekya.models.resnet import Resnet
import torch
from torchvision import transforms
import os
from inclearn.lib.CityscapesClassification import CityscapesClassification
from ekya.drivers.motivation.parser import get_parser

def retraining_run(args):
    NUM_CLASSES = args["num_classes"]
    BATCH_SIZE = args["batch_size"]
    NUM_EPOCHS = args["epochs"]
    LR = args["learning_rate"]
    MOMENTUM = args["momentum"]
    SAVE_PATH = args["checkpoint_path"]
    root = args["root"]  # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    train_sample_names = args["lists_train"].split(',')
    pretrained_sample_names = args["lists_pretrained"].split(',')
    cache = args["use_data_cache"]
    num_tasks = args["num_tasks"]
    train_split = args["train_split"]
    results_path = args["results_path"]
    restore_path = args["restore_path"]
    history_weight = args["history_weight"]


    hyperparams = {'num_hidden': args["num_hidden"],
                            'last_layer_only': not args["disable_last_layer_only"],
                            'model_name': args["model_name"]}

    # Start from pretrained model.
    model = Resnet(NUM_CLASSES, restore_path=restore_path, hyperparameters=hyperparams)

    trsf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    dataset = CityscapesClassification(root, train_sample_names, sample_list_root, transform=trsf, resize_res=224,
                                       use_cache=cache)

    if pretrained_sample_names:
        print("Using pretrained samples in retraining: {}".format(pretrained_sample_names))
        dataset_for_pretrained = CityscapesClassification(root, train_sample_names, sample_list_root, transform=trsf, resize_res=224,
                                           use_cache=cache)
        print("Num samples in pretrained: {}".format(len(dataset_for_pretrained)))
        # dataset.concat_dataset(dataset_for_pretrained)
    dataset_idxs = dataset.samples["idx"]
    num_samples_per_task = int(len(dataset_idxs) / num_tasks)

    results = defaultdict(dict)

    for task in range(num_tasks):
        print("Task {}/{}".format(task, num_tasks))
        # Get train set from previous task
        if task == 0:
            # If first task, no retraining and validation just test
            task_dataset_train_loader = None
            task_dataset_val_loader = None
        else:
            # We include data history too to avoid catastrophic forgetting?
            # task_data_idxs = dataset_idxs[num_samples_per_task * (task - 1):num_samples_per_task * task].values
            task_data_idxs = dataset_idxs[0:num_samples_per_task * task].values
            random.shuffle(task_data_idxs)
            task_data_idxs_train = task_data_idxs[:int(train_split * len(task_data_idxs))]
            task_data_idxs_val = task_data_idxs[int(train_split * len(task_data_idxs)):]

            task_train_dataset = dataset.get_filtered_dataset(task_data_idxs_train)
            task_val_dataset = dataset.get_filtered_dataset(task_data_idxs_val)
            if history_weight > 0:
                # Need to resample the train dataset according to the history_weight param:
                num_new_samples_needed = int(len(dataset_for_pretrained)*(1-history_weight)/history_weight)
                task_train_dataset = task_train_dataset.resample(num_new_samples_needed)
                task_val_dataset = task_val_dataset.resample(num_new_samples_needed)

            if history_weight != 0: # If zero, we do not concat history
                task_train_dataset.concat_dataset(dataset_for_pretrained)
                task_val_dataset.concat_dataset(dataset_for_pretrained)

            task_dataset_train_loader = torch.utils.data.DataLoader(task_train_dataset,
                                                                    batch_size=BATCH_SIZE, shuffle=True,
                                                                    num_workers=args['num_workers'])
            task_dataset_val_loader = torch.utils.data.DataLoader(task_val_dataset,
                                                                  batch_size=BATCH_SIZE, shuffle=True,
                                                                  num_workers=args['num_workers'])

        task_idxs_test = dataset_idxs[num_samples_per_task * task:num_samples_per_task * (task + 1)]
        task_dataset_test = dataset.get_filtered_dataset(task_idxs_test)
        task_dataset_test_loader = torch.utils.data.DataLoader(task_dataset_test, batch_size=BATCH_SIZE, shuffle=True,
                                                               num_workers=args['num_workers'])

        dataloaders_dict = {'train': task_dataset_train_loader,
                            'val': task_dataset_val_loader}

        # Retrain
        print("Retraining")
        _, _, best_val_acc, profile, subprofile = model.train_model(dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR, momentum=MOMENTUM)
        #_, _, best_val_acc = model.train_model(dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR, momentum=MOMENTUM)
        if SAVE_PATH:
            model.save(SAVE_PATH)

        # Testing
        print("Testing")
        test_acc = model.infer(task_dataset_test_loader)

        print("Task {} done.\nVal acc:\t{}\nTest acc:\t{}".format(task, best_val_acc, test_acc))
        results["val_acc"][task] = best_val_acc
        results["test_acc"][task] = test_acc

    results = dict(results)
    print(results)
    with open(results_path, 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    retraining_run(args)