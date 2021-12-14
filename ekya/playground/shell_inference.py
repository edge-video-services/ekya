import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger
import argparse
ray.init()
logger_actor = ray.remote(Logger).remote()
train_model = ray.remote(ICaRLModel).remote(id="1", logger_actor=logger_actor)
infer_model = ray.remote(ICaRLModel).remote(id="2", logger_actor=logger_actor)
result1 = infer_model.restore.remote(path="/tmp/icarl.pt")


ray.get(infer_model.infer.remote(0, high_class_idx=10))
logger_actor.flush.remote()

result = train_model.train.remote(low_class_idx=0, high_class_idx=10, n_epochs=200)