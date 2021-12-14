import itertools
import time

import ray
from inclearn.lib import data

from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

from ekya.utils.monitoractor import Monitor

def get_model(idx, num_epochs, batch_size=None, convnet=None, subsample_dataset=None, learning_rate=None):
    convnet = convnet if convnet else 'resnet50'
    subsample_dataset = subsample_dataset if subsample_dataset else 1
    learning_rate = learning_rate if learning_rate else 2   # default learning rate
    args = {
        'convnet': convnet,
        'lr': learning_rate,
        'scheduling': [int(num_epochs*0.8), int(num_epochs*0.9)]
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, batch_size=batch_size, subsample_dataset=subsample_dataset, logger_actor=logger_actor)

def get_name(*args):
    return ",".join(map(str, args))

def run_trial(batch_size = None, num_epochs = None, freeze_layers = None, subsample_dataset = None, convnet='resnet50', learning_rate=None, num_increments=10):
    batch_size = batch_size if batch_size else 128
    num_epochs = num_epochs if num_epochs else 10
    freeze_layers = freeze_layers if freeze_layers else 0    # good choices for layers: 5,6,7 or 0 for no freezing
    subsample_dataset = subsample_dataset if subsample_dataset else 1   # This is a fraction to reduce dataset size
    learning_rate = learning_rate if learning_rate else 2   # default learning rate
    model_id = get_name(batch_size, num_epochs, learning_rate, subsample_dataset, convnet, num_increments)
    print("Creating model id {}".format(model_id))
    model = get_model(model_id, num_epochs, batch_size, convnet, subsample_dataset, learning_rate)
    if freeze_layers > 0:
        print("Freezing layers {}".format(freeze_layers))
        ray.get(model.freeze_layers.remote(layers_to_freeze=freeze_layers))

    inc_dataset = data.IncrementalDataset(
        dataset_name="cifar100",
        random_order=False,
        shuffle=False,
        batch_size=1,
        workers=10,
        increment=num_increments,
        is_sampleincremental=True
    )
    inc_dataset.new_task_incr()

    for i in range(0, num_increments):
        current_train_idxs = inc_dataset._taskid_to_idxs_map_train[i]
        current_test_idxs = inc_dataset._taskid_to_idxs_map_test[i]
        result = model.train.remote(data_indexes=current_train_idxs, n_epochs=num_epochs, checkpoint_path=None)
        ray.get(result)
        acc_stats = ray.get(model.test_eval.remote(current_test_idxs))
        print("Model {}, task {}, done. Stats: {}".format(model_id, i, acc_stats))
    return acc_stats

if __name__ == '__main__':
    search_space = {
        "batch_size": [16, 32, 256],
        "num_epochs": [20, 50, 70],
        "learning_rate": [2, 8],
        "subsample_dataset": [1],
        "convnet": ["resnet18"],
        "num_increments": [2, 10]
    }
    # search_space = {
    #     "batch_size": [32],
    #     "num_epochs": [1],
    #     "learning_rate": [2],
    #     "subsample_dataset": [1],
    #     "convnet": ["resnet18"],
    #     "num_increments": [2]
    # }

    param_order = list(search_space.keys()) #["batch_size", "num_epochs", "learning_rate", "subsample_dataset", "convnet"]
    hypparam_list = list(itertools.product(*[search_space[param] for param in param_order]))
    ray.init()
    logger_actor = ray.remote(num_cpus=0.01)(Logger).remote()
    monitor_actor = ray.remote(num_cpus=0.01)(Monitor).remote(node_id="node", logger_actor=logger_actor)

    logger_actor.append.remote("trial_config", search_space)
    logger_actor.flush.remote()
    for idx, hypparams in enumerate(hypparam_list):
        param_dict = dict(zip(param_order, hypparams))
        print("Running trial {}/{}.".format(idx, len(hypparam_list)))
        print("Params: {}".format(param_dict))
        run_trial(**param_dict)
    ray.get(logger_actor.flush.remote())
    print("Done.")