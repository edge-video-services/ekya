import itertools
import time

import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

from ekya.utils.monitoractor import Monitor

def get_model(idx, batch_size=None, convnet=None, subsample_dataset=None):
    convnet = convnet if convnet else 'resnet50'
    subsample_dataset = subsample_dataset if subsample_dataset else 1
    args = {
        'convnet': convnet
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, batch_size=batch_size, subsample_dataset=subsample_dataset, logger_actor=logger_actor)

def get_name(batch_size, num_epochs, freeze_layers, dataset, convnet):
    return ",".join(map(str, [batch_size, num_epochs, freeze_layers, dataset, convnet]))

def run_trial(batch_size = None, num_epochs = None, freeze_layers = None, subsample_dataset = None, convnet='resnet50'):
    batch_size = batch_size if batch_size else 128
    num_epochs = num_epochs if num_epochs else 10
    freeze_layers = freeze_layers if freeze_layers else 0    # good choices for layers: 5,6,7 or 0 for no freezing
    subsample_dataset = subsample_dataset if subsample_dataset else 1   # This is a fraction to reduce dataset size
    model_id = get_name(batch_size, num_epochs, freeze_layers, subsample_dataset, convnet)
    print("Creating model id {}".format(model_id))
    model = get_model(model_id, batch_size, convnet, subsample_dataset)
    if freeze_layers > 0:
        print("Freezing layers {}".format(freeze_layers))
        ray.get(model.freeze_layers.remote(layers_to_freeze=freeze_layers))
    result = model.train.remote(low_class_idx=0, high_class_idx=10, n_epochs=num_epochs, checkpoint_path=None)
    ray.get(result)
    acc_stats = ray.get(model.test_eval.remote(low_class_idx=0, high_class_idx=10))
    print("Model {} done. Stats: {}".format(model_id, acc_stats))
    return acc_stats

if __name__ == '__main__':
    search_space = {
        "batch_size": [32, 128, 256],
        "num_epochs": [5, 10, 50],
        "freeze_layers": [0, 5, 7],
        "subsample_dataset": [1],#, 0.5, 0.1],
        "convnet": ["resnet18"]
    }
    param_order = ["batch_size", "num_epochs", "freeze_layers", "subsample_dataset", "convnet"]
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