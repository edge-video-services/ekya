import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICarlTrainer on Ray')
    parser.add_argument('--epochs', type=int, help='Num epochs to train', default=10)
    parser.add_argument('--class-start', type=int, help='Lower range of class idx to train on', default=0)
    parser.add_argument('--class-end', type=int, help='Upper range of class idx to train on', default=10)
    args = parser.parse_args()
    ray.init()
    logger_actor = ray.remote(Logger).remote()
    train_model = ray.remote(ICaRLModel).remote(id="1", logger_actor=logger_actor)
    infer_model = ray.remote(ICaRLModel).remote(id="2", logger_actor=logger_actor)
    result1 = infer_model.restore.remote(path="/tmp/icarl.pt")
    ray.get(infer_model.infer.remote(0, high_class_idx=10))
    result = train_model.train.remote(low_class_idx=args.class_start, high_class_idx=args.class_end, n_epochs=args.epochs)
    ray.get(result)
    result = train_model.checkpoint.remote(path="/tmp/icarl.pt")
    ray.get(result)