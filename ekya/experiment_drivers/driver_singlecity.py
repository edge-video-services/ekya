# Runs one city at a time
import os
from typing import List

import ray

from ekya.classes.camera import Camera
from ekya.classes.ekya import Ekya
from ekya.classes.model import DEFAULT_HYPERPARAMETERS
from ekya.experiment_drivers.arg_parser import get_parser
from ekya.models.hyperparameters import HYPERPARAM_DICT


def create_camera(id, root, num_tasks, train_split, citynames: List[str]) -> Camera:
    return Camera(id=id,
                  train_sample_names=citynames,
                  sample_list_path=os.path.join(root, 'sample_lists', 'citywise'),
                  num_tasks=num_tasks,
                  train_split=train_split)


if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

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
    hyperparam_id = args["hyperparameter_id"]
    dataset_name = args["dataset_name"]

    ray.init(num_cpus=10)

    # Create cameras
    cameras = [Camera(id=str(i),
                      train_sample_names=[c],
                      sample_list_path=os.path.join(root, 'sample_lists', 'citywise'),
                      num_tasks=NUM_TASKS,
                      train_split=train_split,
                      dataset_name=dataset_name,
                      dataset_root=root,
                      pretrained_sample_names=pretrained_cities) for i, c in enumerate(cities)]

    # Set number of epochs
    params = HYPERPARAM_DICT[hyperparam_id]
    params["epochs"] = NUM_EPOCHS
    params['num_classes'] = args['num_classes']
    params['num_hidden'] = args['num_hidden']
    print("Using hyperparameters: {}".format(params))
    scheduler_kwargs = {
        'default_hyperparams': params
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
             termination_task=TERMINATION_TASK
             )
    e.run()
