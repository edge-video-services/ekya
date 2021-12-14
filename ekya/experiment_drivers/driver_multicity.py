# Runs one city at a time
import json
import os
from typing import List

import ray

from ekya.classes.camera import Camera
from ekya.classes.ekya import Ekya
from ekya.experiment_drivers.arg_parser import get_parser
from ekya.microprofilers.modelling_funcs import DEFAULT_SCALED_OPTIMUS_ARGS
from ekya.models.hyperparameters import DEFAULT_HYPERPARAMETERS

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

    model_save_path = args['checkpoint_path']
    golden_model_ckpt = args['golden_model_ckpt_path']
    label_type = "golden_label" if args["golden_label"] else "human"
    NUM_EPOCHS = args["epochs"]
    NUM_TASKS = args["num_tasks"]
    STARTING_TASK = args['start_task']
    TERMINATION_TASK = args['termination_task']
    root = args["root"]  # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    # TODO: LACK OF PRETRAINED MIGHT BE THE REASON FOR REDUCED ACCURACY
    # pretrained_sample_names = args["lists_pretrained"].split(',')
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
    scheduler_name = args["scheduler"]
    hyperparam_id = str(args["hyperparameter_id"])
    dataset_name = args["dataset_name"]
    inference_profile_path = args["inference_profile_path"]
    max_inference_resources = args["max_inference_resources"]

    ray.init(num_cpus=16)
    print("Ray resources: {}".format(ray.available_resources()))

    # Create cameras
    cameras = [Camera(id=str(c),
                      train_sample_names=[c],
                      sample_list_path=os.path.join(root, 'sample_lists', 'citywise'),
                      num_tasks=NUM_TASKS,
                      train_split=train_split,
                      dataset_name=dataset_name,
                      dataset_root=root,
                      pretrained_sample_names=pretrained_cities,
                      inference_profile_path=inference_profile_path,
                      max_inference_resources=max_inference_resources,
                      label_type=label_type,
                      golden_model_ckpt=golden_model_ckpt
                      ) for i, c in enumerate(cities)]

    # Read hyperparams
    with open(args["hyps_path"]) as f:
        HYPERPARAMS = json.load(f)
    params = HYPERPARAMS[hyperparam_id]
    # Set number of epochs for no-retrain and fair scheduler
    params["epochs"] = NUM_EPOCHS
    params['num_classes'] = args['num_classes']
    params['num_hidden'] = args['num_hidden']
    print("Scheduler: {}. Using hyperparameters: {}".format(scheduler_name, params))
    scheduler_kwargs = {
        'default_hyperparams': params
    }
    if scheduler_name.lower() == 'fair':
        # Set the files to read
        scheduler_kwargs = {
            'default_hyperparams': params,
            'inference_weight': args['fair_inference_weight']
        }

    elif scheduler_name.lower() == 'utilitysim':
        # Set the files to read
        scheduler_kwargs = {
            'schedule_path': args["utilitysim_schedule_path"],
            'hyps_path': args["hyps_path"],
            'schedule_key': args["utilitysim_schedule_key"],
            'default_hyperparams': params
        }

    elif scheduler_name.lower() == 'profiling':
        # Set the files to read
        scheduler_kwargs = {
            'profile_hyperparam': params
        }

    elif scheduler_name.lower() == 'thief':
        scheduler_kwargs = {
            'hyps_path': args["hyps_path"],
            'inference_profile_path': inference_profile_path,
            'pretrained_model_dir': pretrained_model_dir,
            'microprofile_device': args['microprofile_device'],
            'microprofile_resources_per_trial': args["microprofile_resources_per_trial"],
            'microprofile_epochs': args["microprofile_epochs"],
            'microprofile_subsample_rate': args["microprofile_subsample_rate"],
            'profiling_epochs': [int(x) for x in args["microprofile_profiling_epochs"].split(',')],
            'default_hyperparams': params,
            'predmodel_acc_args': DEFAULT_SCALED_OPTIMUS_ARGS
        }

    e = Ekya(cameras=cameras,
             scheduler_name=scheduler_name,
             retraining_period=retraining_period,
             num_tasks=NUM_TASKS,
             num_inference_chunks=inference_chunks,
             num_resources=num_gpus,
             scheduler_kwargs=scheduler_kwargs,
             pretrained_model_dir=pretrained_model_dir,
             log_dir=log_dir,
             starting_task=STARTING_TASK,
             termination_task=TERMINATION_TASK,
             profiling_mode=args["profiling_mode"],
             profile_write_path=args['profile_write_path'],
             gpu_memory=args["gpu_memory"]
             )
    e.run()
    print("Ekya is done. Find logs at {}. Terminating.".format(log_dir))
