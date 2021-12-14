import json
import time
from typing import List
import pandas as pd
import numpy as np
import ray

from ekya.classes.camera import Camera
from ekya.classes.model import DEFAULT_HYPERPARAMETERS
from ekya.microprofilers.modelling_funcs import get_scaled_optimus_fn, get_linear_fn
from ekya.microprofilers.runtime_data import MEASURED_TIME_PER_EPOCH, MEASURED_INITTIME
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.microprofilers.simple_microprofiler import SimpleMicroprofiler, subsample_dataloader
from ekya.schedulers.utils import prepare_hyperparameters
from ekya.simulation.camera import generate_training_job
from ekya.simulation.jobs import InferenceJob as SimInferenceJob
from ekya.simulation.schedulers import thief_sco_scheduler
from ekya.utils.dataset_utils import get_pretrained_model_format


class ThiefScheduler(BaseScheduler):
    def __init__(self,
                 hyps_path: str,
                 inference_profile_path: str,
                 pretrained_model_dir: str,
                 microprofile_device='auto',
                 microprofile_resources_per_trial=0.5,
                 microprofile_epochs: int = 5,
                 microprofile_subsample_rate: float = 0.1,
                 profiling_epochs: List[int] = [5, 15, 30],
                 default_hyperparams: dict = DEFAULT_HYPERPARAMETERS,
                 predmodel_acc_args: dict = {}):
        '''
        :param hyps_path: hyp_map.json path which lists the hyperparameter_id to hyperparameter mapping to be used in the scheduler
        Runs the thief scheduler. Includes microprofiling when get_schedule is called.
        '''
        self.pretrained_model_dir = pretrained_model_dir
        self.hyps_path = hyps_path
        self.inference_profile = pd.read_csv(inference_profile_path)
        self.microprofile_device = microprofile_device
        self.microprofile_resources_per_trial = microprofile_resources_per_trial
        self.microprofile_epochs = microprofile_epochs
        self.microprofile_subsample_rate = microprofile_subsample_rate
        self.profiling_epochs = np.array(profiling_epochs)
        self.default_hyperparams = default_hyperparams
        self.predmodel_acc_args = predmodel_acc_args
        with open(self.hyps_path) as f:
            self.hyperparameters = json.load(f)
        for id, hyps in self.hyperparameters.items():
            prepare_hyperparameters(hyps)

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
        # Return fair schedule for inference
        inference_resource_weights = {c.id: (resources / (len(cameras))) * 100 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, hyperparameters

    def get_schedule(self,
                     cameras: Camera,
                     resources: float,
                     state: dict) -> [dict, dict, dict]:
        task_id = state["task_id"]
        retraining_period = state["retraining_period"]

        microprofile_start_time = time.time()
        # Run microprofiling for each camera - both training and inference
        microprofile_results = self.execute_microprofiling(cameras, task_id)
        # Get default inference accuracy from microprofile results
        default_inference_accs = {i: [hp_result['preretrain_test_acc'] for hp_result in result] for i, result in
                                  microprofile_results.items()}  # self.get_default_inference_accs(cameras, task_id, subsample_rate=test_subsample_rate)

        # Generate more profiles by interpolation from micro profiles
        profiles = self.generate_profiles(cameras, microprofile_results, default_inference_accs)

        # Generate SimInferenceJobs
        SimInferenceJobs = {}
        for camera in cameras:
            SimInferenceJobs[camera.id] = SimInferenceJob(
                f"{camera.id}_inference", default_inference_accs[camera.id][0], self.default_hyperparams['model_name'],
                # TODO(romilb): Get inference accuracy from current inference config rather than 0th hyperparam.
                self.inference_profile['subsampling'], self.inference_profile['c1'],
                resource_alloc=0)  # Start with 0 alloc because the scheduler will modify this

        # Generate SimTrainingJobs
        SimTrainingCfgs = {}
        for camera in cameras:
            SimTrainingCfgs[camera.id] = [
                generate_training_job(f"{camera.id}_train_{hp['id']}_{epochs}", acc_prediction, runtime_prediction,
                                      preretrain_acc, model_name=hp['model_name'], oracle=False)
                for [hp, acc_prediction, runtime_prediction, epochs, preretrain_acc] in profiles[camera.id] if
                acc_prediction > preretrain_acc]

        sched_job_pairs = [[SimInferenceJobs[camera.id], SimTrainingCfgs[camera.id]] for camera in cameras]

        microprofile_time_taken = time.time() - microprofile_start_time
        remaining_time = int(retraining_period - microprofile_time_taken)
        assert remaining_time > 0, "Microprofiling took all the time in the retraining period and no retraining " \
                                   "happened. Microprofiling time = {}, Retraining period = {}".format(
            microprofile_time_taken, retraining_period)

        schedule = thief_sco_scheduler(sched_job_pairs,
                                       resources,
                                       remaining_time,
                                       iterations=3,
                                       steal_increment=0.1)
        init_schedule = schedule[0]
        print("[THIEF SCHEDULER] Schedule from thief scheduler: {}".format(init_schedule))
        return self.extract_ekya_schedule(init_schedule, self.hyperparameters)

    @staticmethod
    def extract_ekya_schedule(schedule: dict,
                              hyperparameter_map: dict) -> [dict, dict, dict]:
        '''
        Given a schedule from the thief scheduler, extracts inference_resource_weights, training_resource_weights, hyperparameters to be consumed by ekya.
        :param schedule: From thief scheduler
        :param hyperparameter_map: Map of hp_id to hps
        :return: inference_resource_weights, training_resource_weights, hyperparameters, each dict mapping camera.id to value
        '''
        inference_resource_weights = {}
        training_resource_weights = {}
        hyperparameters = {}
        for job_string, weight in schedule.items():
            components = job_string.split('_')
            if "inference" in job_string:
                camera_id = '_'.join(components[0:-1])
                inference_resource_weights[camera_id] = weight * 100
            elif "train" in job_string:
                epochs = components[-1]
                hp_id = components[-2]
                camera_id = '_'.join(components[0:-3])
                hps = hyperparameter_map[hp_id].copy()
                hps['epochs'] = int(epochs)
                hyperparameters[camera_id] = hps
                training_resource_weights[camera_id] = weight * 100
            else:
                raise Exception("Invalid job string {}. Does not contain inference or train".format(job_string))
        return inference_resource_weights, training_resource_weights, hyperparameters

    def generate_profiles(self, cameras, microprofile_results, default_inference_accs):
        profiles = {}
        unsuccesful_models = 0
        for camera in cameras:
            camera_profiles = []
            for hp_result, default_acc in zip(microprofile_results[camera.id], default_inference_accs[camera.id]):
                test_acc, hyperparameters, init_time, time_per_epoch = hp_result['test_acc'], hp_result[
                    'hyperparameters'], hp_result['init_time'], hp_result['time_per_epoch']
                try:
                    microprofile_accuracy_model = get_scaled_optimus_fn(
                        microprofile_x=np.array([self.microprofile_epochs]),
                        microprofile_y=np.array([test_acc]),
                        start_acc=default_acc,
                        **self.predmodel_acc_args)
                except RuntimeError:
                    unsuccesful_models += 1
                    # Simply return the start accuracy
                    microprofile_accuracy_model = lambda x: default_acc * np.ones_like(x)

                # Get runtime model
                time_per_epoch = MEASURED_TIME_PER_EPOCH[hyperparameters['id']]
                init_time = MEASURED_INITTIME[hyperparameters['id']]
                microprofile_runtime_model = get_linear_fn(a=time_per_epoch,
                                                           b=init_time)

                acc_predictions = microprofile_accuracy_model(self.profiling_epochs)
                runtime_predictions = microprofile_runtime_model(self.profiling_epochs)
                for acc_prediction, runtime_prediction, epochs in zip(acc_predictions, runtime_predictions,
                                                                      self.profiling_epochs):
                    hp_temp = hyperparameters.copy()
                    hp_temp['epochs'] = int(epochs)
                    camera_profiles.append([hp_temp, acc_prediction, runtime_prediction, int(epochs), default_acc])
            profiles[camera.id] = camera_profiles
        if unsuccesful_models:
            print(
                "[THIEF SCHEDULER][WARN] Failed to generate models for {} cameras. Using default inference accuracy.".format(
                    unsuccesful_models))
        return profiles

    def execute_microprofiling(self, cameras, task_id):
        hyp_list = list(self.hyperparameters.values())
        ray_microprof = ray.remote(SimpleMicroprofiler)
        microprofs = {}
        microprof_tasks = {}
        for camera in cameras:
            this_microprof = ray_microprof.options(num_cpus=0).remote(self.microprofile_device)
            microprofs[camera.id] = this_microprof
            dataloaders = [camera._get_dataloader(task_id=task_id, train_batch_size=hp["train_batch_size"],
                                                  test_batch_size=hp["test_batch_size"], subsample_rate=hp["subsample"])
                           for hp in hyp_list]

            pretrained_model_format = get_pretrained_model_format(camera.dataset_name, self.pretrained_model_dir)
            microprof_task = microprofs[camera.id].run_microprofiling.remote(candidate_hyperparams=hyp_list,
                                                                             dataloaders=dataloaders,
                                                                             resources=self.microprofile_resources_per_trial,
                                                                             epochs=self.microprofile_epochs,
                                                                             pretrained_model_format=pretrained_model_format,
                                                                             subsample_rate=self.microprofile_subsample_rate)
            microprof_tasks[camera.id] = microprof_task
        micrprofile_results = {}
        for camera in cameras:
            best_result, results = ray.get(microprof_tasks[camera.id])
            micrprofile_results[
                camera.id] = results  # List of [{best_val_acc, hyperparameters, init_time, time_per_epoch, profile_preretrain_test_acc, profile_test_acc}]
            ray.kill(microprofs[camera.id])  # Kill to free up GPU explicitly
        del microprofs
        return micrprofile_results

    def get_default_inference_accs(self, cameras, task_id, subsample_rate):
        tasks = {}
        for camera in cameras:
            # Get dataloader and subsample it
            dataloaders = camera._get_dataloader(task_id=task_id,
                                                 train_batch_size=self.default_hyperparams["train_batch_size"],
                                                 test_batch_size=self.default_hyperparams["test_batch_size"],
                                                 subsample_rate=subsample_rate)
            subsampled_test_dataloader = subsample_dataloader(dataloaders['test'], subsample_rate)
            inference_model_actor = ray.get_actor("{}_inference".format(camera.id))
            tasks[camera.id] = inference_model_actor.test_acc.remote(test_loader=subsampled_test_dataloader,
                                                                     resource_scaled=False)
        default_inference_accs = {}
        for cid, task in tasks.items():
            default_inference_accs[cid] = ray.get(task)
        return default_inference_accs

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        return fair_reallocation(completed_camera_name,
                                 inference_resource_weights,
                                 training_resources_weights)