# Runs retraining/inference outside of ekya.
import json
import os
from typing import List

import ray

from ekya.classes.camera import Camera
from ekya.classes.ekya import Ekya
from ekya.experiment_drivers.arg_parser import get_parser
from ekya.schedulers import ProfilingScheduler

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

    NUM_EPOCHS = args["epochs"]
    NUM_TASKS = args["num_tasks"]
    STARTING_TASK = args['start_task']
    root = args["root"]  # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    train_split = args["train_split"]
    pretrained_model_dir = args["restore_path"]
    cities = args["cities"].split(',')
    if args["lists_pretrained"] == '':
        pretrained_cities = []
    else:
        pretrained_cities = args["lists_pretrained"].split(',')
    log_dir = args["log_dir"]
    retraining_period = args["retraining_period"]
    inference_chunks = args["inference_chunks"]
    num_gpus = args["num_gpus"]
    hyperparam_id = str(args["hyperparameter_id"])
    dataset_name = args["dataset_name"]
    inference_profile_path = args["inference_profile_path"]
    max_inference_resources = args["max_inference_resources"]
    ray.init(num_cpus=16)

    # Create cameras
    camera = Camera(id=str(cities[0]),
                      train_sample_names=[cities[0]],
                      sample_list_path=os.path.join(root, 'sample_lists', 'citywise'),
                      num_tasks=NUM_TASKS,
                      train_split=train_split,
                      dataset_name=dataset_name,
                      dataset_root=root,
                      pretrained_sample_names=pretrained_cities,
                      inference_profile_path=inference_profile_path,
                      max_inference_resources=max_inference_resources
                      )

    # Read hyperparams
    with open(args["hyps_path"]) as f:
        HYPERPARAMS = json.load(f)
    params = HYPERPARAMS[hyperparam_id]
    # Set number of epochs for no-retrain and fair scheduler
    params["epochs"] = NUM_EPOCHS
    params['num_classes'] = args['num_classes']
    params['num_hidden'] = args['num_hidden']
    dummy_sched = ProfilingScheduler(profile_hyperparam=params)
    params = dummy_sched.prepare_hyperparameters(params)
    print("Using hyperparameters: {}".format(params))

    path_format = "pretrained_cityscapes_fftmunster_{}_{}x2.pt"
    pretrained_model_path = os.path.join(pretrained_model_dir, path_format.format(
        params["model_name"],
        params["num_hidden"]))
    camera.set_current_task(STARTING_TASK)
    task = camera.run_retraining(hyperparameters=params,
                          training_gpu_weight=1,
                          dataloaders_dict={},
                          restore_path=pretrained_model_path,
                          validation_freq=1,
                          profiling_mode=False)

    best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc1, profile_test_acc1 = ray.get(task)


    # Create cameras
    camera = Camera(id=str("2"),
                      train_sample_names=[cities[0]],
                      sample_list_path=os.path.join(root, 'sample_lists', 'citywise'),
                      num_tasks=NUM_TASKS,
                      train_split=train_split,
                      dataset_name=dataset_name,
                      dataset_root=root,
                      pretrained_sample_names=pretrained_cities,
                      inference_profile_path=inference_profile_path,
                      max_inference_resources=max_inference_resources
                      )

    # Read hyperparams
    with open(args["hyps_path"]) as f:
        HYPERPARAMS = json.load(f)
    params = HYPERPARAMS[hyperparam_id]
    # Set number of epochs for no-retrain and fair scheduler
    params["epochs"] = NUM_EPOCHS
    params['num_classes'] = args['num_classes']
    params['num_hidden'] = args['num_hidden']
    dummy_sched = ProfilingScheduler(profile_hyperparam=params)
    params = dummy_sched.prepare_hyperparameters(params)
    print("Using hyperparameters: {}".format(params))

    path_format = "pretrained_cityscapes_fftmunster_{}_{}x2.pt"
    pretrained_model_path = os.path.join(pretrained_model_dir, path_format.format(
        params["model_name"],
        params["num_hidden"]))
    camera.set_current_task(STARTING_TASK)
    task = camera.run_retraining(hyperparameters=params,
                          training_gpu_weight=1,
                          dataloaders_dict={},
                          restore_path=pretrained_model_path,
                          validation_freq=1,
                          profiling_mode=False)

    best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc = ray.get(task)

    print("\n\n")
    print(profile_preretrain_test_acc)
    print(profile_test_acc)

    print("\n\n")
    print(profile_preretrain_test_acc1)
    print(profile_test_acc1)