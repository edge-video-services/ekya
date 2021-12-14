from typing import List

from ekya.classes.camera import Camera


class JobTypes:
    INFERENCE = "inference"
    TRAINING = "training"

def fair_reallocation(completed_camera_name: str,
                      inference_resource_weights: dict,
                      training_resources_weights: dict) -> [dict, dict]:
    '''
    Implements a stateless fair reallocation of resources among inference jobs
    when a training job completes.
    :param completed_camera_name: str, name of the job completed
    :param inference_resource_weights: the current inference resource allocation
    :param training_resources_weights: the current training resource allocation
    :return: new_inference_resource_weights, new_training_resources_weights
    '''
    if completed_camera_name not in training_resources_weights:
        raise KeyError(
            'Completed job {} not found in the current training resource weights {}.'.format(completed_camera_name,
                                                                                             training_resources_weights.keys()))
    relinquished_resources = training_resources_weights[completed_camera_name]
    # Though it isn't necessary, subtract relinquished_resources for accounting
    training_resources_weights[completed_camera_name] -= relinquished_resources
    # Do a fair reallocation
    new_inference_resource_weights = inference_resource_weights.copy()
    new_training_resource_weights = training_resources_weights.copy()
    # Redistribute only among the non-zero jobs
    eligible_recipients = [x for x, old_allocation in new_inference_resource_weights.items() if old_allocation!=0]
    # Check if any inference jobs exist in the first place. If not, return as is
    if len(eligible_recipients) != 0:
        delta = relinquished_resources / len(eligible_recipients)
        for job_id, allocation in new_inference_resource_weights.items():
            if allocation != 0:
                new_inference_resource_weights[job_id] += delta
    return new_inference_resource_weights, new_training_resource_weights

class BaseScheduler(object):
    def __init__(self):
        pass

    def get_schedule(self,
                     cameras: List[Camera],
                     predictions: dict,
                     resources: float,
                     state: dict) -> [dict, dict, dict]:
        '''

        :param cameras:
        :param predictions:
        :param resources:
        :param: state: Any custom state information that needs to be passed to the scheduler. Eg. Task_id.
        :return: inference resource weights, training resource weights, hyperparameters to use
        '''
        pass

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        '''
        This callback is called when a training job completes. This provides the scheduler an opportunity to reconfigure
        resource allocations for jobs. Currently, only changes to to the inference resources are reflected
        (because updating training jobs would require process restarts, an expensive operation).
        :param completed_camera_name: str, name of the job completed
        :param inference_resource_weights: the current inference resource allocation
        :param training_resources_weights: the current training resource allocation
        :return: new_inference_resource_weights, new_training_resources_weights
        '''
        pass

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
        pass