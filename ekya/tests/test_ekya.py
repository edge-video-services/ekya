import os
import random

import ray

from ekya.CONFIG_DATASET import CITYSCAPES_PATH
from ekya.classes.camera import Camera
from ekya.classes.ekya import Ekya
from ekya.classes.model import DEFAULT_HYPERPARAMETERS


def create_test_camera():
    return Camera(id="test_{}".format(random.randint(0, 100)),
                  train_sample_names=['zurich'],
                  sample_list_path=os.path.join(CITYSCAPES_PATH, 'sample_lists', 'citywise'),
                  num_tasks=10,
                  train_split=0.8)


if __name__ == '__main__':
    ray.init(num_cpus=10)
    NUM_CAMERAS = 1
    NUM_TASKS = 5
    NUM_EPOCHS = 15
    cameras = [create_test_camera() for x in range(0, NUM_CAMERAS)]
    params = DEFAULT_HYPERPARAMETERS
    params["epochs"] = NUM_EPOCHS
    scheduler_kwargs = {
        'default_hyperparams': params
    }
    e = Ekya(cameras=cameras,
             scheduler_name='noretrain',
             retraining_period=120,
             num_tasks=NUM_TASKS,
             num_inference_chunks=10,
             num_resources=1,
             scheduler_kwargs=scheduler_kwargs,
             pretrained_model_dir='/home/romilb/research/msr/models'
             )
    e.run()
