import os
import ray

from ekya.CONFIG_DATASET import CITYSCAPES_PATH
from ekya.classes.camera import Camera
from ekya.classes.ekya import inference_executor
from ekya.classes.model import DEFAULT_HYPERPARAMETERS

if __name__ == '__main__':
    ray.init()
    c = Camera(id="me",
               train_sample_names=['zurich'],
               sample_list_path= os.path.join(CITYSCAPES_PATH, 'sample_lists', 'citywise'),
               num_tasks=10,
               train_split=0.8)
    c.increment_task()
    c.increment_task()
    c.update_inference_model(hyperparameters=DEFAULT_HYPERPARAMETERS,
                             inference_gpu_weight=1,
                             restore_path="")
    executor_task = inference_executor.remote(c,
                                              hyperparameters=DEFAULT_HYPERPARAMETERS,
                                              inference_gpu_weight=1,
                                              num_chunks=10, retraining_period=100, test_batch_size=32)
    print(ray.get(executor_task))