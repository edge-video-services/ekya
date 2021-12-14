from typing import List

from ekya.classes.camera import Camera
from ekya.classes.model import DEFAULT_HYPERPARAMETERS
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation


class ProfilingScheduler(BaseScheduler):
    def __init__(self,
                 profile_hyperparam=DEFAULT_HYPERPARAMETERS,
                 validation_freq = 3):
        '''
        Profiling scheduler which does not run inference and allocates resources only to retraining.
        Uses a fixed set of hyperparameters
        :param profile_hyperparam: Hyperparameter to profile
        :param validation_freq: Validation frequency to use if not specified in hyperparameters
        '''
        self.validation_freq = validation_freq
        self.profile_hyperparam = self.prepare_hyperparameters(profile_hyperparam)
        pass

    def prepare_hyperparameters(self,
                                hyps: dict) -> dict:
        if "train_batch_size" not in hyps:
            hyps["train_batch_size"] = hyps["batch_size"] // 8
        if "test_batch_size" not in hyps:
            hyps["test_batch_size"] = DEFAULT_HYPERPARAMETERS["test_batch_size"]
        if "num_classes" not in hyps:
            hyps["num_classes"] = DEFAULT_HYPERPARAMETERS["num_classes"]
        if "validation_freq" not in hyps:
            hyps["validation_freq"] = self.validation_freq
        return hyps

    def get_schedule(self,
                     cameras: List[Camera],
                     resources: float,
                     state: dict) -> [dict, dict, dict]:
        inference_resource_weights = {c.id: 0 for c in cameras}    # 0 for no inference
        training_resource_weights = {c.id: (resources/(len(cameras)))*100 for c in cameras}
        hyperparameters = {c.id: self.profile_hyperparam for c in cameras}
        return inference_resource_weights, training_resource_weights, hyperparameters

    def get_inference_schedule(self,
                                cameras: List[Camera],
                                resources: float):
        '''
        Returns the schedule when inference only jobs must be run. This must be super fast since this is the schedule
        used before the get_schedule actual schedule is obtained.
        :param cameras: list of cameras
        :param resources: total resources in the system to be split across tasks
        :return: inference resource weights, hyperparameters
        '''
        inference_resource_weights = {c.id: 0 for c in cameras}
        hyperparameters = {c.id: self.profile_hyperparam for c in cameras}
        return inference_resource_weights, hyperparameters

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        return fair_reallocation(completed_camera_name,
                          inference_resource_weights,
                          training_resources_weights)