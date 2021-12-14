import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger
import argparse

def get_model(idx):
    args = {
        'convnet': 'resnet50'
    }
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(args=args, id=idx, logger_actor=logger_actor, subsample_dataset=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICarlTrainer on Ray')
    parser.add_argument('--epochs', type=int, help='Num epochs to train', default=100)
    parser.add_argument('--class-start', type=int, help='Lower range of class idx to train on', default=0)
    parser.add_argument('--class-end', type=int, help='Upper range of class idx to train on', default=10)
    args = parser.parse_args()
    ray.init()
    logger_actor = ray.remote(Logger).remote()
    train_model = get_model("1")
    result = train_model.train.remote(low_class_idx=args.class_start, high_class_idx=args.class_end, n_epochs=args.epochs)
    ray.get(result)
    result = train_model.checkpoint.remote(path="/tmp/icarl_resnet50_0_10.pt")
    ray.get(result)