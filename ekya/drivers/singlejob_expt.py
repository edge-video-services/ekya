import random
import time

import ray
from inclearn.lib import data

from ekya.models.icarl_model import ICaRLModel
from ekya.models.inference_manager import InferenceManager
from ekya.utils.loggeractor import Logger
import argparse
import numpy as np

from ekya.utils.monitoractor import Monitor


def get_model(idx, batch_size, lr, checkpoint_path = None):
    args = {
        'convnet': 'resnet18',
        'learning_rate': lr
    }
    model_actor = ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, logger_actor=logger_actor, batch_size=batch_size, subsample_dataset=1)
    if checkpoint_path:
        print("Restoring from {}".format(checkpoint_path))
        model_actor.restore.remote(path=checkpoint_path)
    return model_actor

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser(description='ICarl Infer + Trainer on Ray')
    parser.add_argument('--epochs', type=int, help='Num epochs to train', default=70)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
    parser.add_argument('--lr', type=float, help='Learning rate', default=2)
    parser.add_argument('--start-checkpoint-path', type=str, help='Path to restore initial model from',
                        default="/tmp/icarl_start.pt")
    parser.add_argument('--save-checkpoint-path', type=str, help='Path to save', default="/tmp/icarl_expt.pt")
    parser.add_argument('--duration', type=int, help='experiment duration', default=600)
    parser.add_argument('--warmup-duration', type=int, help='warmup duration', default=60)
    parser.add_argument('--num-incr', type=int, help='Number of increments to train on', default=1)
    args = parser.parse_args()
    args = vars(args)
    ray.init()
    logger_actor = ray.remote(num_cpus=0)(Logger).remote()
    monitor_actor = ray.remote(num_cpus=0)(Monitor).remote(node_id="node", logger_actor=logger_actor)

    infer_model = get_model("1", batch_size=args["batch_siz"
                                                 "e"], lr=args["lr"], checkpoint_path=args["start_checkpoint_path"])
    infer_manager = ray.remote(num_cpus=0)(InferenceManager).remote("1", infer_model,logger_actor)
    ray.get(infer_manager.register_handle.remote(infer_manager))
    train_model = get_model("2", batch_size=args["batch_size"], lr=args["lr"], checkpoint_path=args["start_checkpoint_path"])

    # Get indexes for training on new data
    NUM_INCREMENTS = 10
    inc_dataset = data.IncrementalDataset(
        dataset_name="cifar100",
        random_order=False,
        shuffle=True,
        batch_size=1,
        workers=0,
        increment=NUM_INCREMENTS,
        is_sampleincremental=True
    )
    inc_dataset.new_task_incr()
    # current_train_idxs = inc_dataset._taskid_to_idxs_map_train[1]   # THIS IS IMPORTANT - train on the next iteration
    current_train_idxs = np.concatenate(list(inc_dataset._taskid_to_idxs_map_train.values())[1:args["num_incr"]+1])  # Use this to combine multiple increments
    current_test_idxs = inc_dataset._taskid_to_idxs_map_test[0]

    # Start inference and let it run for 60s to establish starting accuracy
    inference_task = infer_manager.run_inference_loop.remote()
    time.sleep(args["warmup_duration"])

    # Start training
    ray.get(infer_manager.set_contention.remote(True))
    training_task = train_model.train.remote(current_train_idxs, n_epochs=args["epochs"])
    start_time = time.time()

    # Wait for train to finish
    ray.get([training_task])
    ray.get(infer_manager.set_contention.remote(False))

    # Checkpoint the train and load in the inference model
    ray.get(train_model.checkpoint.remote(args["save_checkpoint_path"]))
    ray.get(infer_manager.restore.remote(args["save_checkpoint_path"]))

    # Inference should keep running automatically, run it
    current_time = time.time()
    while (current_time - start_time < args["duration"]):
        time.sleep(1)
        current_time = time.time()
        print("Training is done, waiting for retrain period to end.")

    print("Done running experiment!")
    logger_actor.flush.remote()