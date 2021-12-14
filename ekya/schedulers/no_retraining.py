from ekya.classes.camera import Camera
from ekya.classes.model import DEFAULT_HYPERPARAMETERS
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.schedulers.utils import prepare_hyperparameters


class NoRetrainingScheduler(BaseScheduler):
    def __init__(self,
                 default_hyperparams=DEFAULT_HYPERPARAMETERS):
        '''
        Scheduler which does not do any retraining.
        :param default_hyperparams:
        '''
        self.default_hyperparams = prepare_hyperparameters(default_hyperparams)

    def get_schedule(self,
                     cameras: Camera,
                     resources: float,
                     state: dict) -> [dict, dict, dict]:
        inference_resource_weights = {c.id: (resources/len(cameras))*100 for c in cameras}
        training_resource_weights = {c.id: 0 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, training_resource_weights, hyperparameters

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        return fair_reallocation(completed_camera_name,
                          inference_resource_weights,
                          training_resources_weights)