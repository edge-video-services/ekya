import time

import ray
from ray.experimental import signal

from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

#TODO - This example is incompelete
from ekya.utils.monitoractor import Monitor

def get_model(idx):
    args = {
        'convnet': 'resnet50'
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, logger_actor=logger_actor)

N_EPOCHS = 3

if __name__ == '__main__':
    ray.init()
    init_tasks = []
    checkpoint_restore_times = []
    checkpoint_write_time = 0
    id = 0
    logger_actor = ray.remote(Logger).remote()
    monitor_actor = ray.remote(Monitor).remote(node_id="node", logger_actor=logger_actor)

    models = []
    for _ in range(0, 30):
        id+=1
        print("Attempting to launch model id {}".format(id))
        logger_actor.append.remote("model_count", [time.time(), id])
        logger_actor.flush.remote()
        new_model = get_model(id)
        models.append(new_model)
        ray.get(new_model.set_handle.remote(new_model))    # Wait for initialization.

        print("model id {}: Model added".format(id))
        tasks = []
        for model in models:
            tasks.append(model.train_atom.remote(0, 10, N_EPOCHS))
        try:
            ray.get(tasks)
        except:
            print("Launch failed.")
            logger_actor.append.remote("signal_fail_time", [time.time()])
            break
        print("Launch of model id {} successful.".format(id))


def get_total():
    total = 0
    for name, param in mlarge.named_parameters():
        if param.requires_grad:
            total +=1
            print(name)
    print(total)