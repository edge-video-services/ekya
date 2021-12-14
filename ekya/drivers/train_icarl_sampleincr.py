import ray
from inclearn.lib import data

from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger
import argparse
import numpy as np

def get_model(idx, batch_size, lr):
    args = {
        'convnet': 'resnet18',
        'learning_rate': lr
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, logger_actor=logger_actor, batch_size=batch_size, subsample_dataset=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICarlTrainer on Ray')
    parser.add_argument('--epochs', type=int, help='Num epochs to train', default=70)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
    parser.add_argument('--lr', type=float, help='Learning rate', default=2)
    parser.add_argument('--checkpoint-path', type=str, help='Path to save', default="/tmp/icarl_resnet18_sampleincr_0.1data.pt")
    parser.add_argument('--restore-path', type=str, help='Path to restore from', default=None)
    parser.add_argument('--infer-only', help='Run inference only', action="store_true", default=False)
    args = parser.parse_args()
    args = vars(args)
    ray.init()
    logger_actor = ray.remote(Logger).remote()
    train_model = get_model("1", batch_size=args["batch_size"], lr=args["lr"])

    # Train on first 50 samples for each class
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
    current_train_idxs = inc_dataset._taskid_to_idxs_map_train[0]
    current_test_idxs = inc_dataset._taskid_to_idxs_map_test[0]

    if args["restore_path"]:
        print("Restoring from {}".format(args["restore_path"]))
        ray.get(train_model.restore.remote(path=args["restore_path"]))
        print("Restored, setting new train samples")
        current_train_idxs = np.concatenate(list(inc_dataset._taskid_to_idxs_map_train.values())[1:])
    print("Model train on samples {}, started.".format(len(current_train_idxs)))
    if not args["infer_only"]:
        result = ray.get(train_model.train.remote(current_train_idxs, n_epochs=args["epochs"]))
    acc_stats = ray.get(train_model.test_eval.remote(current_test_idxs))
    print("Model train on samples {}, done. Stats: {}".format(len(current_train_idxs), acc_stats))

    result = train_model.checkpoint.remote(path=args["checkpoint_path"])
    ray.get(result)