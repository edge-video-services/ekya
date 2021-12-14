import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

if __name__ == '__main__':
    ray.init()
    logger_actor = Logger()
    model1 = ray.remote(ICaRLModel).remote(id="1", logger_actor=logger_actor)
    model2 = ray.remote(ICaRLModel).remote(id="2", logger_actor=logger_actor)
    result1 = model1.restore.remote(path="/tmp/icarl.pt")
    result2 = model2.restore.remote(path="/tmp/icarl.pt")
    ray.get([result1, result2])
    result = model1.infer.remote(0, high_class_idx=10)
    ray.get(result)