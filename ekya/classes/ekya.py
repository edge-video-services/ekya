import json
import os
import time
from collections import defaultdict

import ray
from typing import List

from ray.exceptions import RayActorError

from ekya.CONFIG import RANDOM_SEED, DEFAULT_GPU_MEMORY
from ekya.CONFIG_DATASET import PRETRAINED_MODEL_DIR
from ekya.schedulers.utils import convert_to_ray_demands, quantize_demands
from ekya.utils.dataset_utils import get_pretrained_model_format
from ekya.utils.helpers import seed_all
from ekya.utils.loggeractor import Logger
from torch.utils.data import DataLoader

from ekya.classes.camera import Camera
from ekya.schedulers import get_scheduler

seed_all(RANDOM_SEED)

@ray.remote
def inference_executor(camera: Camera,
                       num_chunks: int,
                       retraining_period: int,
                       test_batch_size: int,
                       logger: ray.actor.ActorHandle = None) -> None:
    # WARNING: Camera object is frozen copy at method invocation. Updates to camera wont be reflected in the remote function
    # Parse dataloader and break it into num_chunks
    test_loader = camera._get_dataloader(camera.current_task, test_batch_size=test_batch_size)["test"]
    test_dataset = test_loader.dataset
    index = test_dataset.get_indexes()
    chunk_size = len(test_dataset)//num_chunks
    time_per_chunk = retraining_period//num_chunks
    print("[InferenceExecutor] Initializing inference executor for camera {}. Total samples in task: {}. Chunk size: "
          "{}. Time per chunk: {}".format(camera.id, len(test_dataset), chunk_size, time_per_chunk))

    test_accs = []

    def fetch_actor():
        inference_model_actor = None
        while inference_model_actor is None:
            try:
                inference_model_actor = ray.get_actor("{}_inference".format(camera.id))
            except ValueError:
                print("Got a dead actor, retrying..")
                time.sleep(0.5)
        return inference_model_actor

    #Iterate over each chunk and run it's inference
    for chunk_id in range(num_chunks):
        start_time = time.time()
        chunk_idxs = index[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
        chunk_dataset = test_dataset.get_filtered_dataset(chunk_idxs)
        chunk_loader = DataLoader(chunk_dataset,
                                              batch_size=test_loader.batch_size,
                                              shuffle=True,
                                              num_workers=test_loader.num_workers)
        # Try to fetch the latest actor handle from ray until successful
        inference_model_actor = fetch_actor()
        retry_count = 0
        test_acc = None
        while test_acc is None and retry_count < 5:
            try:
                test_acc = ray.get(inference_model_actor.test_acc.remote(test_loader=chunk_loader))
            except:
                retry_count+=1
                print("[InferenceExecutor][WARNING] Test_acc task failed. Retrying. Attempt: {}".format(retry_count))
                inference_model_actor = fetch_actor()
                test_acc = None
        print("[InferenceExecutor] {}: Got test acc: {}".format(camera.id, test_acc))
        test_accs.append(test_acc)

        # Result logging
        if logger:
            data = [time.time(), camera.current_task, chunk_id, test_acc]
            logger.append.remote("inference_{}".format(camera.id), data)

        end_time = time.time()
        chunk_remaining_time = time_per_chunk - (end_time - start_time)
        if chunk_remaining_time > 0:
            print("Chunk {}/{} done. Sleeping for {:.2f}".format(chunk_id, num_chunks, chunk_remaining_time))
            time.sleep(chunk_remaining_time)
        else:
            # TODO: Log warning: Throughput is less than line rate.
            print("Warning: Inference xput is less than line rate in chunk {}, camera {}".format(chunk_id, camera.id))
    return test_accs

class Ekya(object):
    def __init__(self,
                 cameras: List[Camera],
                 scheduler_name: str,
                 retraining_period: int,
                 num_tasks: int,
                 num_inference_chunks: int,
                 num_resources: float,
                 pretrained_model_dir:str,
                 log_dir: str = "",
                 scheduler_kwargs: dict = {},
                 starting_task: int = 0,
                 termination_task: int = -1,
                 profiling_mode: bool = False,
                 profile_write_path: str = '',
                 gpu_memory: float = DEFAULT_GPU_MEMORY):
        '''
        The class to encapsulate the system. Contains methods to setup and run the inference/training in a a loop over multiple retraining windows.
        A new retraining window is trigerred whenever the time elapsed since the last trigger is greater than retraining period.
        :param cameras: List of Camera objects in the system
        :param scheduler_name: Scheduler to use
        :param retraining_period: Length of retraining period in seconds
        :param num_tasks: Number of retraining periods to create. The dataset will be split into size/num_periods sized chunks.
        :param num_inference_chunks: Number of chunks to break the inference data per window into.
        :param num_resources: Number of GPUs/resources available for scheduling
        :param scheduler_kwargs: Custom arguments for the scheduler class.
        :param starting_task: Task id to start at.
        :param termination_task: Task id to end at. If -1, then all tasks are run.
        :param profiling_mode: Whether to log profiles of training jobs
        :param profile_write_path: Path where to log the profiles
        :param gpu_memory: Memory available per GPU
        '''
        self.cameras = cameras
        scheduler_class = get_scheduler(scheduler_name)
        self.scheduler = scheduler_class(**scheduler_kwargs)
        self.retraining_period = retraining_period
        self.num_tasks = num_tasks
        self.num_inference_chunks = num_inference_chunks
        self.current_task = starting_task-1 # -1 because not started yet.
        self.termination_task = termination_task if termination_task > 0 else self.num_tasks - 1
        self.last_retraining_start_time = 0
        self.num_resources = num_resources
        self.pretrained_model_dir = pretrained_model_dir
        self.profiling_mode = profiling_mode
        self.profile_write_path = profile_write_path
        self.gpu_memory = gpu_memory

        # Setup profiling datastructures, if necessary:
        if self.profiling_mode:
            os.makedirs(profile_write_path, exist_ok=True)
            # Keys for these dicts: camera_id. Values: {task: <dict returned from resnet.py>}
            nested_dict = lambda: defaultdict(nested_dict)
            self.profiling_profiling_results = nested_dict()
            self.profiling_subprofiling_test_results = nested_dict()
            self.profiling_overall_results = nested_dict()
            self.profiling_metadata = nested_dict()


        # Launch logger actor
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.logger = ray.remote(Logger).options(num_cpus=0).remote(base_dir=log_dir)

    def stop_current_jobs(self):
        '''
        Stops all current jobs in the system. Used at the start of a retraining.
        :return:
        '''
        # TODO: Log results if any are still running.
        print("Stopping all jobs.")
        for task in self.inference_tasks.values():
            ray.cancel(task, force=True)
        for camera in self.cameras:
            if hasattr(camera, "training_model"):
                ray.kill(camera.training_model)
                del camera.training_model
            if hasattr(camera, "inference_model"):
                ray.kill(camera.inference_model)
                del camera.inference_model
            # WARN: This will result in RayActorErrors for some of the get tasks - they must be handled in the calling functions.

    def update_inference_jobs(self,
                              inference_resource_weights: dict,
                              hyperparameters: dict,
                              ray_inference_resource_demands: dict,
                              blocking: bool = False):
        # Updates inference weights and launches inference exectuoir if not already running
        for camera in self.cameras:
            this_inference_weight = inference_resource_weights.get(camera.id, 0)
            camera.inference_gpu_weight = this_inference_weight
            if this_inference_weight > 0:
                this_inference_ray_demand = ray_inference_resource_demands[camera.id]
                camera.inference_ray_demand = this_inference_ray_demand
                # Set the weights on the inference job and load the pretrained default model
                pretrained_model_path = get_pretrained_model_format(camera.dataset_name, self.pretrained_model_dir).format(
                                                                  hyperparameters[camera.id]["model_name"],
                                                                  hyperparameters[camera.id]["num_hidden"])
                camera.update_inference_model(hyperparameters[camera.id],
                                              this_inference_weight,
                                              this_inference_ray_demand,
                                              restore_path=pretrained_model_path,
                                              blocking=blocking)
                if camera.id not in self.inference_tasks:
                    # Run inference executor if not already running.
                    self.inference_tasks[camera.id] = inference_executor.remote(camera,
                                                                                num_chunks=self.num_inference_chunks,
                                                                                retraining_period=self.retraining_period,
                                                                                test_batch_size=hyperparameters[camera.id]["test_batch_size"],
                                                                                logger=self.logger)
            elif this_inference_weight == 0:
                print("[WARN][Task {}] Camera {} was assigned 0 or no resources for inference. Not running inference".format(
                    camera.current_task, camera.id))


    def launch_training_jobs(self,
                             training_resource_weights: dict,
                             hyperparameters: dict,
                             ray_training_resource_demands: dict):
        # Launch retraining tasks if it is not the 0th task
        if self.current_task > 0:
            for camera in self.cameras:
                this_train_weight = training_resource_weights.get(camera.id, 0)
                camera.training_gpu_weight = this_train_weight
                # Run retraining only if some resource weight is allocated
                if this_train_weight > 0:
                    this_training_ray_demand = ray_training_resource_demands[camera.id]
                    camera.training_ray_demand = this_training_ray_demand
                    pretrained_model_path = get_pretrained_model_format(camera.dataset_name, self.pretrained_model_dir).format(
                                                                  hyperparameters[camera.id]["model_name"],
                                                                  hyperparameters[camera.id]["num_hidden"])
                    if self.profiling_mode:
                        model_save_dir = os.path.join(self.profile_write_path, str(camera.id))
                        os.makedirs(model_save_dir, exist_ok=True)
                    else:
                        model_save_dir = ""
                    (self.retraining_tasks[camera.id],
                     self.retraining_metadata[camera.id]) = camera.run_retraining(hyperparameters[camera.id],
                                                                         training_resource_weights[camera.id],
                                                                         this_training_ray_demand,
                                                                         dataloaders_dict={},
                                                                         restore_path=pretrained_model_path,
                                                                         profiling_mode=self.profiling_mode,
                                                                         model_save_dir=model_save_dir)
                if this_train_weight == 0:
                    print("[Task {}] Camera {} was assigned 0 or no resources for retraining. Not retraining.".format(camera.current_task, camera.id))

    def get_memory_demands(self, cameras=None):
        # Returns memory demands as a fraction of per GPU memory
        if not cameras:
            cameras = self.cameras
        inference_memory_demands = {}
        training_memory_demands = {}
        for camera in cameras:
            inference_memory_demands[camera.id] = camera.inference_memory_footprint()/self.gpu_memory
            training_memory_demands[camera.id] = camera.training_memory_footprint()/self.gpu_memory
        return inference_memory_demands, training_memory_demands

    def new_task_callback(self):
        '''
        This method increments tasks and computes a new schedule and runs tasks for execution with the right resource allocations.
        :return:
        '''
        # Increment tasks
        self.current_task += 1
        print("New task callback called. Now on task {}/{}".format(self.current_task, self.num_tasks))
        for camera in self.cameras:
            camera.set_current_task(self.current_task)

        self.inference_tasks = {}
        self.retraining_tasks = {}
        self.retraining_metadata = {}
        self.inference_resource_weights, self.training_resource_weights = {}, {}

        # Launch inference jobs as the clock starts ticking even before the scheduler is done
        self.inference_resource_weights, self.current_hyperparameters = self.scheduler.get_inference_schedule(self.cameras, self.num_resources)
        self.inference_memory_demand, self.training_memory_demand = self.get_memory_demands(self.cameras)

        # Get ray resource demands to launch inference jobs and quantize for packing
        ray_inference_resource_demands, _ = convert_to_ray_demands(self.inference_memory_demand, self.inference_memory_demand,
                                                                                               self.training_memory_demand, self.training_memory_demand)    # HACK: Reduced demand to let micrprofiling run
        ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
        print("[Ekya] Inference Scheduler allocation: Inference: {}. Ray demands: {}\n".format(self.inference_resource_weights, ray_inference_resource_demands))
        self.update_inference_jobs(self.inference_resource_weights, self.current_hyperparameters, ray_inference_resource_demands)

        # Future work: These resource allocations should be over time. Consquently, resource allocations must be updated over time.
        custom_state = {"task_id": self.current_task,
                        "retraining_period": self.retraining_period}
        prev_hyperparams = self.current_hyperparameters
        self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters = self.scheduler.get_schedule(self.cameras, self.num_resources, custom_state)    # {camera.id: schedule/hyperparameters}

        # If resources were not allocated for training, preserve previous hyperparameters
        for camera in self.cameras:
            if camera.id not in self.current_hyperparameters:
                self.current_hyperparameters[camera.id] = prev_hyperparams[camera.id]

        # Get ray resource demands and quantize for packing
        ray_inference_resource_demands, ray_training_resource_demands = convert_to_ray_demands(self.inference_memory_demand, self.inference_resource_weights,
                                                                   self.training_memory_demand, self.training_resource_weights)
        ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
        ray_training_resource_demands = quantize_demands(ray_training_resource_demands)
        print("[Ekya] Training+Inference Scheduler allocation: Training: {}\nInference: {}\n Ray Training: {}\n Ray Inference: {}".format(self.training_resource_weights, self.inference_resource_weights, ray_training_resource_demands, ray_inference_resource_demands))
        self.log_schedules(self.current_task, self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters)

        # Update inference jobs with new weights from the scheduler and launch retraining jobs
        self.update_inference_jobs(self.inference_resource_weights, self.current_hyperparameters, ray_inference_resource_demands)
        self.launch_training_jobs(self.training_resource_weights, self.current_hyperparameters, ray_training_resource_demands)

    def log_schedules(self, task_id, inference_resource_weights, training_resource_weights, current_hyperparameters):
        # log_format = [task_id, job_type, camera_name, weight, hp_id, hp_epochs]
        for camera_name, weight in training_resource_weights.items():
            job_type = 'training'
            hp_id = current_hyperparameters[camera_name]["id"]
            hp_epochs = current_hyperparameters[camera_name]["epochs"]
            log_data = [task_id, job_type, camera_name, weight, hp_id, hp_epochs]
            self.logger.append.remote("schedule", log_data)
        for camera_name, weight in inference_resource_weights.items():
            job_type = 'inference'
            hp_id = current_hyperparameters[camera_name]["id"]
            hp_epochs = current_hyperparameters[camera_name]["epochs"]
            log_data = [task_id, job_type, camera_name, weight, hp_id, hp_epochs]
            self.logger.append.remote("schedule", log_data)


    def accumulate_profiles(self, camera_id, task_id):
        retraining_task = self.retraining_tasks[camera_id]
        best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_return = ray.get(retraining_task)
        metadata = self.retraining_metadata[camera_id]
        hyps = self.current_hyperparameters[camera_id]
        hparam_id = str(hyps['id'])

        self.profiling_profiling_results[camera_id][hparam_id][task_id] = profile
        self.profiling_subprofiling_test_results[camera_id][hparam_id][task_id] = subprofile_test_results
        self.profiling_overall_results[camera_id][hparam_id]["val_acc"][task_id] = best_val_acc
        self.profiling_overall_results[camera_id][hparam_id]["test_acc"][task_id] = profile_test_acc
        self.profiling_overall_results[camera_id][hparam_id]["preretrain_test_acc"][task_id] = profile_preretrain_test_acc
        self.profiling_metadata[camera_id][hparam_id][task_id] = metadata

    def flush_profiles(self):
        for c in self.cameras:
            camera_id = c.id
            results_path = os.path.join(self.profile_write_path, camera_id)
            os.makedirs(results_path, exist_ok=True)

            # Write hyperparam results
            for hparam_idx, result in self.profiling_overall_results[camera_id].items():
                hyp_result_path = os.path.join(results_path, "{}_retraining_result.json".format(hparam_idx))
                with open(hyp_result_path, 'w') as fp:
                    json.dump(result, fp)

                # Write hyperparam profile
                hyp_profile_path = os.path.join(results_path, "{}_profile.json".format(hparam_idx))
                with open(hyp_profile_path, 'w') as fp:
                    json.dump(self.profiling_profiling_results[camera_id][hparam_idx], fp)

                # Write subprofile test results
                hyp_subprofile_path = os.path.join(results_path, "{}_subprofile_test.json".format(hparam_idx))
                with open(hyp_subprofile_path, 'w') as fp:
                    json.dump(self.profiling_subprofiling_test_results[camera_id][hparam_idx], fp)
                # Write metadata
                hyp_metadata_path = os.path.join(
                    results_path, "{}_metadata.json".format(hparam_idx))
                with open(hyp_metadata_path, 'w') as fp:
                    json.dump(self.profiling_metadata[camera_id][hparam_idx], fp)

    def retraining_completion_callback(self, camera_id):
        if self.profiling_mode:
            # Accumulate profiles for all tasks into a dict which is written to disk at the end.
            current_task = str(self.current_task)
            self.accumulate_profiles(camera_id, current_task)


    def check_task_loop(self):
        # A method to check the currently running training tasks and if complete, load their model onto the inference task
        print("Starting check task loop.")
        running_retraining_tasks = list(self.retraining_tasks.values())
        while True:
            remaining_time = self.retraining_period - (time.time() - self.last_retraining_start_time)
            if self.profiling_mode and not running_retraining_tasks:
                # Terminate task when profiling completes.
                return
            if remaining_time > 0:
                done_tasks, running_retraining_tasks = ray.wait(running_retraining_tasks, timeout=0)
                retraining_results = ray.get(done_tasks)
                done_camera_ids = [next(camera_id for camera_id, task_id in self.retraining_tasks.items() if task_id == t) for t in done_tasks]
                done_cameras = [next(c for c in self.cameras if c.id == i) for i in done_camera_ids]

                # Retrained accuracy logging
                if self.logger:
                    log_time = time.time()
                    for i, [best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_results] in enumerate(retraining_results):
                        retraining_time_taken = log_time - self.last_retraining_start_time
                        camera = done_cameras[i]
                        hp_id = self.current_hyperparameters[camera.id]["id"]
                        hp_epochs = self.current_hyperparameters[camera.id]["epochs"]
                        data = [log_time, camera.current_task, retraining_time_taken, best_val_acc, profile_preretrain_test_acc,
                                profile_test_acc, hp_id, hp_epochs]
                        self.logger.append.remote("retraining_{}".format(camera.id), data)

                for c in done_cameras:
                    if self.current_task > 0:   # 0th task has no retraining
                        if True:
                            self.retraining_completion_callback(c.id)
                            self.inference_resource_weights, self.training_resource_weights = \
                                self.scheduler.reallocation_callback(c.id,
                                                                     self.inference_resource_weights,
                                                                     self.training_resource_weights)
                            ray_inference_resource_demands, _ = convert_to_ray_demands(self.inference_memory_demand,
                                                                                       self.inference_resource_weights,
                                                                                       self.training_memory_demand,
                                                                                       self.training_resource_weights)
                            ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
                            # Update the inference model with new retrained weights if inference is running
                            if c.inference_gpu_weight > 0:
                                c.update_inference_from_retrained_model()
                            # Update remaining cameras' inference weights
                            # NOTE(ROMILB): Updating each inference job's weights through restarts might be expensive
                            # NOTE(ROMILB): Consider parallelizing this.
                            for x in self.cameras:
                                # Update only if inference was running (wt > 0)
                                if x.id != c.id and self.inference_resource_weights[x.id] > 0:
                                    x.update_inference_model(self.current_hyperparameters[x.id],
                                                             self.inference_resource_weights[x.id],
                                                             ray_inference_resource_demands[x.id],
                                                             blocking=False)
                        else:
                            print("[WARN] Not doing anything on train job completion.")
                print("Check task loop: remaining time: {}".format(remaining_time))
                time.sleep(1)   # Sleep before checking again. WARN: This might cause timing leaks.
            if remaining_time <= 0:
                return


    def run(self):
        '''
        Main loop for Ekya.
        :return:
        '''
        done = False # This is set to true to exit the loop. Happens when current_task > num_tasks
        while not done:
            print("Started loop")
            this_loop_start_time = time.time()
            self.last_retraining_start_time = this_loop_start_time
            # TODO: Make this async:
            self.new_task_callback()

            # Check running retraining tasks and load their models into inference when they complete.
            self.check_task_loop()  # This loop terminates when the time period for the task is done.

            # ray.get([self.inference_tasks[c.id] for c in self.cameras])
            # Stop all jobs. Required to stop any rogue jobs running
            self.stop_current_jobs()
            if self.current_task == self.termination_task:   # We have completed the termination task so end here.
                self.logger.flush.remote()
                if self.profiling_mode:
                    self.flush_profiles()
                done = True
