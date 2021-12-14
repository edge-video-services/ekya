# Generate data for fig 1
import json
import random
from collections import defaultdict

from ekya.models.resnet import Resnet
import torch
from torchvision import transforms
import os
from inclearn.lib.CityscapesClassification import CityscapesClassification
from ekya.drivers.motivation.parser import get_parser
import itertools
import numpy as np

HYPERPARAM_SPACE = None

HYPERPARAM_LIST = [{'num_hidden': 512, 'last_layer_only': True, 'learning_rate': 0.001, 'model_name': "resnet50",
                    'batch_size': 64, 'subsample': 1, 'momentum': 0.9},
                   ]

EXPERIMENTS = ['no_retrain', 'retrain_5tasks', 'dummy', 'continuous_retrain']    # Retrains every 5 tasks


def generate_hyperlist_from_space(hyperparam_space):
    if hyperparam_space is not None:
        return list(dict(zip(hyperparam_space, x)) for x in itertools.product(*hyperparam_space.values()))
    else:
        return HYPERPARAM_LIST


def retrainingfreq_run(args):
    NUM_CLASSES = args["num_classes"]
    NUM_EPOCHS = args["epochs"]
    NUM_SUBPROFILES = args["num_subprofiles"]
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
    train_from_scratch = not args["do_not_train_from_scratch"]  # Default is true, so it will not train from scratch and carry forward the previous model. Specify -dtfs to train from custom model

    hyperparams_list = generate_hyperlist_from_space(HYPERPARAM_SPACE)


    trsf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    dataset = CityscapesClassification(root, train_sample_names, sample_list_root, transform=trsf, resize_res=224,
                                       use_cache=cache)
    if pretrained_sample_names:
        print("Using pretrained samples in retraining: {}".format(pretrained_sample_names))
        dataset_for_pretrained = CityscapesClassification(root, train_sample_names, sample_list_root, transform=trsf,
                                                          resize_res=224,
                                                          use_cache=cache)
    dataset_idxs = dataset.samples["idx"]
    num_samples_per_task = int(len(dataset_idxs) / num_tasks)

    print("New sample list size: {}. Tasks: {}. Samples per task: {}".format(
        len(dataset_idxs),
        num_tasks,
        num_samples_per_task
    ))

    results = defaultdict(lambda: defaultdict(dict))    # Stores accuracy values in the old format
    profiling_results = defaultdict(dict)
    subprofiling_test_results = defaultdict(dict)

    best_model_weights = {}    # Hashed by num_hidden or other restore-constraining hash
    best_model_accs = {}

    # Since we need to test on all tasks, generate a map of taskid to loaders before execution:
    test_loaders = {}   # Map of taskid to it's test loader
    for task in range(num_tasks):
        task_idxs_test = dataset_idxs[num_samples_per_task * task:num_samples_per_task * (task + 1)]
        task_dataset_test = dataset.get_filtered_dataset(task_idxs_test)
        test_loaders[task] = torch.utils.data.DataLoader(task_dataset_test, batch_size=32, shuffle=True,
                                                               num_workers=args['num_workers'])
    ep_mult = int(NUM_EPOCHS/NUM_SUBPROFILES)
    test_epochs = [x-1 for x in list(range(ep_mult, NUM_EPOCHS+1, ep_mult))]    # List of subprofile epochs where the model must be tested on all tasks. -1 because indexed at 0.
    print("Num subprofiles: {}. Will test the model at epochs: {}.".format(NUM_SUBPROFILES, test_epochs))

    for EXPT in EXPERIMENTS:
        for task in range(num_tasks):
            for hparam_idx, hyperparam in enumerate(hyperparams_list):
                num_hidden = hyperparam['num_hidden']
                LR = hyperparam["learning_rate"]
                MOMENTUM = hyperparam["momentum"]
                BATCH_SIZE = hyperparam["batch_size"]
                subsample = hyperparam["subsample"]
                model_name = hyperparam["model_name"]

                model_hash = str(model_name) + "_" + str(num_hidden)

                temp_best_model_weights = {}
                cp_path = restore_path + "{}_{}x2.pt".format(model_name, num_hidden) if restore_path else restore_path
                model = Resnet(NUM_CLASSES, restore_path=cp_path, hyperparameters=hyperparam)

                print("Task {}/{}, hyperparam {}/{}".format(task, num_tasks,
                                                            hparam_idx, len(hyperparams_list)))

                if model_hash in best_model_weights and not train_from_scratch:
                    print("Not training from scratch, reloading previous task's model dict.")
                    model.model.load_state_dict(best_model_weights[model_hash])

                # Setup test dataset
                task_idxs_test = dataset_idxs[num_samples_per_task * task:num_samples_per_task * (task + 1)]
                task_dataset_test = dataset.get_filtered_dataset(task_idxs_test)
                task_dataset_test_loader = torch.utils.data.DataLoader(task_dataset_test, batch_size=BATCH_SIZE,
                                                                       shuffle=True,
                                                                       num_workers=args['num_workers'])
                # For first task, no retraining and validation just test
                task_dataset_train_loader = None
                task_dataset_val_loader = None

                if task >= 1:
                    # If not first task, run retraining and validation and test
                    if EXPT == 'continuous_retrain':
                        # We include data history too to avoid catastrophic forgetting?
                        # task_data_idxs = dataset_idxs[num_samples_per_task * (task - 1):num_samples_per_task * task].values
                        task_data_idxs = dataset_idxs[0:num_samples_per_task * task].values

                        # Shuffle and Subsample dataset
                        random.shuffle(task_data_idxs)
                        print("Subsample: {}".format(subsample))
                        task_data_idxs = np.random.choice(task_data_idxs, int(len(task_data_idxs)*subsample), replace=False)

                        pretrained_subsample_idxs = dataset_for_pretrained.samples["idx"].values
                        pretrained_subsample_idxs = np.random.choice(pretrained_subsample_idxs, int(len(pretrained_subsample_idxs) * subsample), replace=False)
                        pretrained_subsampled_dataset = dataset_for_pretrained.get_filtered_dataset(pretrained_subsample_idxs)

                        task_data_idxs_train = task_data_idxs[:int(train_split * len(task_data_idxs))]
                        task_data_idxs_val = task_data_idxs[int(train_split * len(task_data_idxs)):]

                        task_train_dataset = dataset.get_filtered_dataset(task_data_idxs_train)
                        task_train_dataset.concat_dataset(pretrained_subsampled_dataset)
                        task_dataset_train_loader = torch.utils.data.DataLoader(task_train_dataset,
                                                                                batch_size=BATCH_SIZE, shuffle=True,
                                                                                num_workers=args['num_workers'])

                        task_val_dataset = dataset.get_filtered_dataset(task_data_idxs_val)
                        task_val_dataset.concat_dataset(pretrained_subsampled_dataset)
                        task_dataset_val_loader = torch.utils.data.DataLoader(task_val_dataset,
                                                                              batch_size=BATCH_SIZE, shuffle=True,
                                                                              num_workers=args['num_workers'])

                        print("Subsampling done. Task {} train data: {}, val data: {}, pretrain_data: {}".format(task, len(task_train_dataset), len(task_val_dataset), len(pretrained_subsampled_dataset)))

                    if EXPT == 'retrain_5tasks':
                        if task >= 5:   # Retrain at 5th task:
                            print("Preparing data for task 5.")
                            # We include data history too to avoid catastrophic forgetting?
                            # task_data_idxs = dataset_idxs[num_samples_per_task * (task - 1):num_samples_per_task * task].values
                            task_data_idxs = dataset_idxs[0:num_samples_per_task * 5].values    # Dirty hack, instead maintain history

                            # Shuffle and Subsample dataset
                            # Pick num_samples_per_task from the past tasks
                            task_data_idxs = np.random.choice(task_data_idxs, num_samples_per_task, replace=False)

                            pretrained_subsample_idxs = dataset_for_pretrained.samples["idx"].values
                            pretrained_subsample_idxs = np.random.choice(pretrained_subsample_idxs,
                                                                         int(len(pretrained_subsample_idxs) * subsample),
                                                                         replace=False)
                            pretrained_subsampled_dataset = dataset_for_pretrained.get_filtered_dataset(
                                pretrained_subsample_idxs)

                            task_data_idxs_train = task_data_idxs[:int(train_split * len(task_data_idxs))]
                            task_data_idxs_val = task_data_idxs[int(train_split * len(task_data_idxs)):]

                            task_train_dataset = dataset.get_filtered_dataset(task_data_idxs_train)
                            task_train_dataset.concat_dataset(pretrained_subsampled_dataset)
                            task_dataset_train_loader = torch.utils.data.DataLoader(task_train_dataset,
                                                                                    batch_size=BATCH_SIZE, shuffle=True,
                                                                                    num_workers=args['num_workers'])

                            task_val_dataset = dataset.get_filtered_dataset(task_data_idxs_val)
                            task_val_dataset.concat_dataset(pretrained_subsampled_dataset)
                            task_dataset_val_loader = torch.utils.data.DataLoader(task_val_dataset,
                                                                                  batch_size=BATCH_SIZE, shuffle=True,
                                                                                  num_workers=args['num_workers'])

                            print("Subsampling done. Task {} train data: {}, val data: {}, pretrain_data: {}".format(task, len(task_train_dataset), len(task_val_dataset), len(pretrained_subsampled_dataset)))

                dataloaders_dict = {'train': task_dataset_train_loader,
                                    'val': task_dataset_val_loader,
                                    'test': task_dataset_test_loader}    #NOTE: Adding test every epoch because profiling

                # Setup subprofiling test dict:
                subprofile_test_epochs = {e: test_loaders for e in test_epochs}

                # Retrain
                print("Retraining")
                _, _, best_val_acc, profile, subprofile_test_results = model.train_model(dataloaders_dict,
                                                                                         subprofile_test_epochs = subprofile_test_epochs,
                                                                                         num_epochs=NUM_EPOCHS,
                                                                                         lr=LR,
                                                                                         momentum=MOMENTUM)
                if SAVE_PATH:
                    model.save(SAVE_PATH)

                # Testing
                print("Testing")
                test_acc = model.infer(task_dataset_test_loader)    # This is the only number we really care about.

                print("Task {} done.\nVal acc:\t{}\nTest acc:\t{}".format(task, best_val_acc, test_acc))
                results[hparam_idx]["val_acc"][task] = best_val_acc
                results[hparam_idx]["test_acc"][task] = test_acc
                profiling_results[hparam_idx][task] = profile
                subprofiling_test_results[hparam_idx][task] = subprofile_test_results

                # Save the model weights if it has the best accuracy but dont commit yet
                best_acc = best_model_accs.get(model_hash, 0)
                if test_acc > best_acc:
                    best_model_accs[model_hash] = test_acc
                    temp_best_model_weights[model_hash] = model.model.state_dict()
            best_model_weights = temp_best_model_weights.copy()

        results = dict(results)
        print(results)
        expt_result_path = os.path.join(results_path, EXPT)
        os.makedirs(expt_result_path, exist_ok=True)
        for hparam_idx, result in results.items():
            # Write hyperparam results
            hyp_result_path = os.path.join(expt_result_path, "{}_retraining_result.json".format(hparam_idx))
            with open(hyp_result_path, 'w') as fp:
                json.dump(result, fp)

            # Write hyperparam profile
            hyp_profile_path = os.path.join(expt_result_path, "{}_profile.json".format(hparam_idx))
            with open(hyp_profile_path, 'w') as fp:
                json.dump(profiling_results[hparam_idx], fp)

            # Write subprofile test results
            hyp_subprofile_path = os.path.join(expt_result_path, "{}_subprofile_test.json".format(hparam_idx))
            with open(hyp_subprofile_path, 'w') as fp:
                json.dump(subprofiling_test_results[hparam_idx], fp)

        # Write the hyperparam mapping
        hyp_map_path = os.path.join(expt_result_path, "hyp_map.json")
        hyp_map = {idx: hyp for idx, hyp in enumerate(hyperparams_list)}
        with open(hyp_map_path, 'w') as fp:
            json.dump(hyp_map, fp)

        # Write the hyperparam space
        hyp_space_path = os.path.join(expt_result_path, "hyp_space.json")
        with open(hyp_space_path, 'w') as fp:
            json.dump(HYPERPARAM_SPACE, fp)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    retrainingfreq_run(args)
