import os
import ray

from ekya.CONFIG_DATASET import CITYSCAPES_PATH
from ekya.classes.camera import Camera
from ekya.classes.model import DEFAULT_HYPERPARAMETERS

if __name__ == '__main__':
    ray.init()
    c = Camera(id="me",
               train_sample_names=['zurich'],
               sample_list_path= os.path.join(CITYSCAPES_PATH, 'sample_lists', 'citywise'),
               num_tasks=10,
               train_split=0.8)
    #c._get_dataloader(0,10)
    c.increment_task()
    c.increment_task()
    task = c.run_retraining(hyperparameters=DEFAULT_HYPERPARAMETERS,
                     training_gpu_weight=1)
    print(ray.get(task))