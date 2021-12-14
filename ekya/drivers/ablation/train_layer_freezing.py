import time

import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

from ekya.utils.monitoractor import Monitor

def get_model(idx):
    args = {
        'convnet': 'resnet50'
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, logger_actor=logger_actor)

TRAIN_EPOCHS = 40
if __name__ == '__main__':
    ray.init()
    init_tasks = []
    checkpoint_restore_times = []
    checkpoint_write_time = 0
    logger_actor = ray.remote(num_cpus=0.01)(Logger).remote()
    monitor_actor = ray.remote(num_cpus=0.01)(Monitor).remote(node_id="node", logger_actor=logger_actor)
    num_infer_models = 1
    num_train_models = 1
    infer_models = [get_model(i) for i in range (1, num_infer_models+1)]

    init_tasks.extend([infer_model.set_handle.remote(infer_model) for infer_model in infer_models])
    init_tasks.extend([infer_model.restore.remote(path="/tmp/icarl_resnet50_0_10.pt") for infer_model in infer_models])
    ray.get(init_tasks)

    # Run inference on the first task
    infer_result = []
    for i in range(0,10):
        infer_result.extend([infer_model.infer.remote(0, high_class_idx=20) for infer_model in infer_models])
    ray.get(infer_result)
    print("Task 1 Inference Result: {}".format(infer_result))


    [infer_model.infer_loop.remote(0, 20) for infer_model in infer_models]
    # Start retrain on the second task and keep the inference running
    init_tasks = []
    train_baseline_models = [get_model(i) for i in range(num_infer_models+1, num_infer_models + 1 + num_train_models)]
    init_tasks.extend([train_baseline_model.restore.remote(path="/tmp/icarl_resnet50_0_10.pt") for train_baseline_model in train_baseline_models])
    ray.get(init_tasks)
    baseline_train_futures = [train_baseline_model.train.remote(low_class_idx=10, high_class_idx=20, n_epochs=TRAIN_EPOCHS, checkpoint_interval=0) for train_baseline_model in train_baseline_models]
    print("Waiting for baseline completion")
    ray.wait(baseline_train_futures)

    del train_baseline_models
    train_frozen_models = [get_model(i) for i in range(num_infer_models + 1 + num_train_models, num_infer_models + 1 + num_train_models + num_train_models)]
    init_tasks = []
    init_tasks.extend(
        [train_frozen_model.restore.remote(path="/tmp/icarl_resnet50_0_10.pt") for train_frozen_model in
         train_frozen_models])
    init_tasks.extend(
        [train_frozen_model.freeze_layers.remote(layers_to_freeze = 7) for train_frozen_model in
         train_frozen_models])
    ray.get(init_tasks)

    frozen_train_futures = [
        frozen_baseline_model.train.remote(low_class_idx=10, high_class_idx=20, n_epochs=TRAIN_EPOCHS, checkpoint_interval=0) for
        frozen_baseline_model in train_frozen_models]

    print("Waiting for frozen completion")
    ray.wait(frozen_train_futures)

    print("Running 6 layer frozen")
    del train_frozen_models
    del frozen_train_futures
    train_frozen_models = [get_model(i) for i in range(num_infer_models + 1 + num_train_models*2,
                                                       num_infer_models + 1 + num_train_models*3)]
    init_tasks = []
    init_tasks.extend(
        [train_frozen_model.restore.remote(path="/tmp/icarl_resnet50_0_10.pt") for train_frozen_model in
         train_frozen_models])
    init_tasks.extend(
        [train_frozen_model.freeze_layers.remote(layers_to_freeze=6) for train_frozen_model in
         train_frozen_models])
    ray.get(init_tasks)

    frozen_train_futures = [
        frozen_baseline_model.train.remote(low_class_idx=10, high_class_idx=20, n_epochs=TRAIN_EPOCHS, checkpoint_interval=0) for
        frozen_baseline_model in train_frozen_models]

    print("Running 5 layer frozen")
    ray.wait(frozen_train_futures)

    del train_frozen_models
    del frozen_train_futures
    train_frozen_models = [get_model(i) for i in range(num_infer_models + 1 + num_train_models*3,
                                                       num_infer_models + 1 + num_train_models*4)]
    init_tasks = []
    init_tasks.extend(
        [train_frozen_model.restore.remote(path="/tmp/icarl_resnet50_0_10.pt") for train_frozen_model in
         train_frozen_models])
    init_tasks.extend(
        [train_frozen_model.freeze_layers.remote(layers_to_freeze=5) for train_frozen_model in
         train_frozen_models])
    ray.get(init_tasks)

    frozen_train_futures = [
        frozen_baseline_model.train.remote(low_class_idx=10, high_class_idx=20, n_epochs=TRAIN_EPOCHS, checkpoint_interval=0) for
        frozen_baseline_model in train_frozen_models]
    ray.wait(frozen_train_futures)
    logger_actor.flush.remote()
    print("Done.")