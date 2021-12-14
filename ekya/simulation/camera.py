"""This module defines the camera used in Ekya simulator."""
import numpy as np

from ekya.simulation.jobs import InferenceJob, TrainingJob
from ekya.simulation.constants import OPT


# Functions to model training accuracy/inference accuracy vs. resource begin


def slowed_acc(acc, contention_slowdown=0.9):
    return acc * contention_slowdown


def profile_fn(t, profile_time, profile_acc):
    # Returns a continuous function from the given profile time and acc values.
    return np.interp(t, profile_time, profile_acc)


def optimus_fn(t, T, K=1):
    if t == 0:
        return 0
    else:
        # Hits 0.9 acc at time T. *100 for accuracy.
        return 1 / ((1 / (t * 10 * 1 / T)) + 1) * 100 * K


def inv_optimus_fn(a, T, K=1):  # Maps accuracy to time
    if a == 0:
        return 0
    else:
        return a / ((10 / T) * (100*K - a))


def k_optimus_fn(a, T, t):
    # Returns the value of k for a given a, T and t
    # solve a=1 / ((1 / (t * 10 * 1 / (t+d))) + 1) * 100 * K,
    # b=1 / ((1 / ((t+d) * 10 * 1 / (t+d))) + 1) * 100 * K for K, t
    return ((a*T) / 10*t + a) / 100


def linear_fn(t, k):
    return min(k * t, 1)


def inv_linear_fn(a, k):
    return a / k


def tanh_fn(t, scale_factor, shift=0):
    return (np.tanh(((t - shift) * 2 * scale_factor - scale_factor)) + 1) / 2


def inv_tanh_fn(p, k):
    (k - np.arctanh(1 - 2 * p)) / 2 * k


def get_linear_fn(k):
    return lambda t: linear_fn(t, k), lambda a: inv_linear_fn(a, k)


def get_optimus_fn(T, K=1):
    return lambda t: optimus_fn(t, T, K), lambda a: inv_optimus_fn(a, T, K)


def get_tanh_fn(k):
    return lambda t: tanh_fn(t, k), lambda a: inv_tanh_fn(a, k)

# Functions to model training accuracy/inference accuracy vs. resource end


def generate_training_job(name, final_accuracy, job_time, start_accuracy,
                          model_name, inference_job=None, oracle=False):
    if start_accuracy > final_accuracy:
        if oracle:
            print(f"WARNING: {name} model is already trained. start_accuracy "
                  f"{start_accuracy} > final_accuracy {final_accuracy}. "
                  "Still adding..")
        else:
            raise Exception(f"WARNING: {name} model is already trained. "
                            f"start_accuracy {start_accuracy} > final_accuracy"
                            f" {final_accuracy}. This is invalid.")
    init_time = 0

    if OPT:
        # a linear modeling training accuracy changing vs. time.
        def func(x):
            return final_accuracy if x >= job_time else (
                x * (final_accuracy - start_accuracy) / (
                    job_time - init_time) + start_accuracy)
    else:
        def func(x): return np.interp(
            x, [init_time, job_time], [start_accuracy, final_accuracy])
    return TrainingJob(name, func, init_time, job_time, resource_alloc=0,
                       model_name=model_name, inference_job=inference_job)


def generate_training_job2(name, final_accuracy, job_time, model_name,
                           start_accuracy=50, inference_job=None):
    # Starts from given accuracy
    if start_accuracy > final_accuracy:
        raise Exception("The model is already trained. start_accuracy "
                        f"{start_accuracy} > final_accuracy {final_accuracy}")
    # solve a=1 / ((1 / (t * 10 * 1 / (t+d))) + 1) * 100 * K, \
    # b=1 / ((1 / ((t+d) * 10 * 1 / (t+d))) + 1) * 100 * K for K, t
    a = start_accuracy
    b = final_accuracy
    d = job_time
    k = 11 * b / 1000
    t = a * d / (1000 * k - 11 * a)
    conv_time = t + d

    func, inv_func = get_optimus_fn(conv_time, k)
    init_time = inv_func(start_accuracy)
    return TrainingJob(name, func, init_time, conv_time, resource_alloc=0,
                       model_name=model_name, inference_job=inference_job)


class Camera(object):
    """Camera class used to simulate a camera."""

    def __init__(self, name, taskwise_training_profiles,
                 taskwise_oracle_profiles, taskwise_start_profiles,
                 inference_job=None, subsampling=None,
                 inference_camera_profile=None, start_task_id="1"):
        """Initialize Camera object.

        Args
            name(str): name of a camera.
            inference_job(InferenceJob)
            taskwise_start_accuracy(dict): {task_id: accs}.
                WARNING: How is the acc determined? Which Inference profile is
                it?  That's preselected by the user right now..
            taskwise_training_profiles(dict): {task_id: profge iles}. Each
                profile is a list of 4 tuples
                [(final_acc, resource_time, start_acc, model_name), ..]
            taskwise_start_profiles(dict): {task_id: profge iles}. Each
                profile is a list of 4 tuples
                [(final_acc, resource_time, start_acc, model_name), ..]
            taskwise_oracle_profiles(dict): {task_id: profge iles}. Each
                profile is a list of 4 tuples
                [(final_acc, resource_time, start_acc, model_name), ..]
            subsampling(pandas series): a list of subsampling rate.
            inference_camera_profile(pandas series): inference profile with
                respect to subsampling.
            start_task_id(str): the id of start task

        """
        self.name = name
        self.taskwise_training_profiles = taskwise_training_profiles
        self.taskwise_oracle_profiles = taskwise_oracle_profiles
        self.taskwise_start_profiles = taskwise_start_profiles
        self.current_accuracy = taskwise_start_profiles[start_task_id][0]
        # the model used by camera at the start of inference is consistent
        # with the current accuracy
        model_name = taskwise_start_profiles[start_task_id][1]
        self.train_configs = []  # list of all training jobs from prediction
        self.oracle_configs = []  # list of all training jobs from oracle
        if inference_job:
            self.inference_job = inference_job
        else:
            self.inference_job = InferenceJob(
                f"{self.name}_inference", self.current_accuracy, model_name,
                subsampling, inference_camera_profile, resource_alloc=0)

    def get_training_configurations(self):
        return self.train_configs

    def generate_training_configurations(self, task_id):
        """Generate the valid training configurations based on prediction.

        Only create training job when predicted final acc > current_acc.

        Args
            task_id(str)

        Return
            a list of training jobs
        """
        configs = []
        # TODO(romilb): Currently not bothering with profile curve, just start
        # and end acc.
        for final_acc, resource_time, hp_id, model_name in \
                self.taskwise_training_profiles[task_id]:
            # Format of hp_id is hpid_epochs
            name = "{}_train_{}".format(self.name, hp_id)
            if final_acc < self.current_accuracy:
                print(f"WARNING(predictedcfg): Final accuracy {final_acc} less"
                      f" than current acc {self.current_accuracy} for camera "
                      f"{self.name}, ignoring profile {name}.")
            else:
                configs.append(generate_training_job(
                    name, final_acc, resource_time, self.current_accuracy,
                    model_name, self.inference_job))
                # no inference job because this is used just for scheduling
        self.train_configs = configs
        return self.train_configs

    def get_oracle_configurations(self):
        return self.oracle_configs

    def generate_oracle_configurations(self, task_id):
        """Generate the valid training configurations based on oracle profile.

        Only create training job when predicted final acc > current_acc.

        Args
            task_id(str)

        Return
            a list of training jobs
        """
        configs = []
        # TODO(romilb): Currently not bothering with profile curve, just start
        # and end acc.
        for final_acc, resource_time, hp_id, model_name in \
                self.taskwise_oracle_profiles[task_id]:
            if final_acc < self.current_accuracy:
                print(f"WARNING(oracle): Final accuracy {final_acc} less than"
                      f" current acc {self.current_accuracy} for camera "
                      f"{self.name}, still adding profile if the predictor "
                      "decides to use it..")
            # No else because the profiles must be added regardless.
            # Format of hp_id is hpid_epochs
            name = "{}_train_{}".format(self.name, hp_id)
            configs.append(generate_training_job(
                name, final_acc, resource_time, self.current_accuracy,
                model_name, self.inference_job, oracle=True))
        self.oracle_configs = configs
        return self.oracle_configs

    def get_inference_job(self):
        return self.inference_job

    # This function is never used so leave this commented for now.
    # def set_current_accuracy(self, acc):
    #     self.current_accuracy = acc
    #     self.inference_job.set_base_accuracy(self.current_accuracy)

    def reset_current_accuracy(self, task_id):
        # Sets accuracy to accuracy of given task
        self.current_accuracy = self.taskwise_start_profiles[task_id][0]
        model_name = self.taskwise_start_profiles[task_id][1]
        self.inference_job.set_base_accuracy(self.current_accuracy)
        if self.inference_job.model_name != model_name:
            self.inference_job.update_model_name(model_name)
