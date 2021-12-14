from typing import List

from ekya.classes.camera import Camera
from ekya.classes.model import DEFAULT_HYPERPARAMETERS
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.schedulers.utils import prepare_hyperparameters


class FairScheduler(BaseScheduler):
    def __init__(self,
                 default_hyperparams=DEFAULT_HYPERPARAMETERS,
                 inference_weight=0.5):
        '''
        Fair scheduler which equally allocates resources to all cameras.
        :param default_hyperparams:
        :param inference_weight: Static weight to assign to inference resources
        '''
        self.inference_weight = inference_weight
        self.default_hyperparams = prepare_hyperparameters(default_hyperparams)

    def get_schedule(self,
                     cameras: Camera,
                     resources: float,
                     state: dict) -> [dict, dict, dict]:
        inference_resource_budget = resources * self.inference_weight
        training_resource_budget = resources - inference_resource_budget
        inference_resource_weights = {c.id: (inference_resource_budget/len(cameras))*100 for c in cameras}    # /2 to split inference and training
        training_resource_weights = {c.id: (training_resource_budget/len(cameras))*100 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, training_resource_weights, hyperparameters

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        return fair_reallocation(completed_camera_name,
                          inference_resource_weights,
                          training_resources_weights)

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
        inference_resource_weights = {c.id: (resources / (len(cameras))) * 100 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, hyperparameters