"""This module defines the jobs used in Ekya simulator."""
import numpy as np

from ekya.simulation.constants import INFINITY, INSTA_CHECKPOINT, INFER_MAX_RES_DICT


def get_infer_profile(subsampling, cam_profile, max_inference_resources=1):
    """Interpolate between subsamping and corresponded accuracy scale.

    Args
        subsampling(pandas series): a list of subsampling rate.
        camera_profile(pandas series): inference profile with respect to
            subsampling.
        max_inference_resources(float): maximum inference resources needed.

    Return
        A lambda function which maps gpu resource to accuracy scale.
        None

    """
    # effective_subsample_rate is max_inference_resources/res + 0.0000001 to
    # avoid divide by zero.
    # max_inference_resources/np.interp(acc_scale, data[camera],
    # data['subsampling'])
    return lambda res: np.interp(
        max_inference_resources/(res+0.000001), subsampling, cam_profile,
        right=0), lambda acc_scale: NotImplementedError


class EkyaJob(object):
    """The base class of different jobs in the simulator."""

    def __init__(self, name, resource_alloc):
        self.name = name
        self.current_resource_alloc = resource_alloc
        self.done = False

    def set_resource_alloc(self, resource_alloc):
        """Update gpu resource allocated to the job."""
        self.current_resource_alloc = resource_alloc

    def is_done(self):
        """Return the done flag. True if the job finishes. Otherwise, False."""
        return self.done

    def __str__(self):
        return self.name


class TrainingJob(EkyaJob):
    """The class which simulates training job."""

    def __init__(self, name, train_acc_vs_t_function, init_train_duration,
                 total_train_duration, resource_alloc, model_name,
                 inference_job=None):
        """Initialize a training job.

        Args
            name(str): name of the training job.
            train_acc_vs_t_function(function): a function describes how the
                trained accuracy changes with respect to training time
                t(seconds). t is measured on a gpu(full resources).
            init_train_duration(float): training time used before the start of
                the training job.
            total_train_duration(float): total training time needed to complete
                the training job.
            resource_alloc(float): gpu resource allocated.
            model_name(str): model name. e.g. resnet18.
            inference_job(InferenJob): the inference job which the training job
                will update.
        """
        self.train_fn = train_acc_vs_t_function  # GPU Cycles
        self.init_train_duration = init_train_duration
        self.trained_duration = self.init_train_duration
        self.acc = self.train_fn(self.trained_duration)
        self.inference_job = inference_job
        self.total_train_duration = total_train_duration
        self.job_name = name
        self.model_name = model_name
        super(TrainingJob, self).__init__(self.job_name, resource_alloc)

    # def reset(self):
    #     self.trained_duration = self.init_train_duration
    #     self.acc = self.train_fn(self.trained_duration)
    #     super(TrainingJob, self).__init__(self.job_name, resource_alloc=0)

    def step(self, wall_time):
        """Run training with specified wall time.

        Args
            wall_time(float): wall time in seconds.

        Return
            Accuracy of the training job after training with wall time.
        """
        if wall_time != 0 and not self.is_done():
            # old_acc = self.acc
            res_time = wall_time * self.current_resource_alloc
            self.trained_duration += res_time
            self.acc = self.train_fn(self.trained_duration + res_time)
            if self.inference_job:
                if INSTA_CHECKPOINT:
                    self.inference_job.set_base_accuracy(self.acc)
                    if self.inference_job.model_name != self.model_name:
                        # give inference job new model name
                        self.inference_job.update_model_name(self.model_name)
            if self.total_train_duration <= self.trained_duration:
                self.done = True
                if not INSTA_CHECKPOINT and self.inference_job:
                    self.inference_job.set_base_accuracy(self.acc)
                    # give inference job new model name
                    if self.inference_job.model_name != self.model_name:
                        self.inference_job.update_model_name(self.model_name)
            # delta_acc = self.acc - old_acc
        elif (wall_time != 0 and self.current_resource_alloc != 0):
            print(f"Warning: Job {self.name} done but still allocated "
                  f"resources {self.current_resource_alloc} and being stepped."
                  " Doing nothing..")
        return self.acc

    def get_accuracy(self):
        return self.acc

    def completion_time(self, allocation_over_walltime):
        """Compute the time needed to complete the job.

        This method answers "If the job is given allocation_over_time
        resources for the remaining duration, how much time will it take to
        complete?"

        Args
            allocation_over_walltime(dict): {time: alloc} eg. {0: 0.25, 10: 1}.

        Return
            time remained(float)
        """
        total_remaining_res_time = (
            self.total_train_duration - self.init_train_duration)
        res_change_times = sorted(allocation_over_walltime.keys())
        for i in range(0, len(res_change_times)):
            remaining_time = res_change_times[i]
            allocation = allocation_over_walltime[res_change_times[i]]
            if i == len(res_change_times) - 1:
                # Last resource allocation, compute directly from this
                if allocation == 0:
                    remaining_time = INFINITY
                else:
                    remaining_time += total_remaining_res_time / allocation
                break
            else:
                block_duration = res_change_times[i + 1] - res_change_times[i]
                if allocation != 0:
                    if total_remaining_res_time/allocation < block_duration:
                        # The job will complete in this allocation block.
                        # Compute the total wall time
                        remaining_time += total_remaining_res_time / allocation
                        break
                    else:
                        # We'll need to run for this allocation and see if it
                        # completes in the next allocation
                        total_remaining_res_time -= block_duration * allocation
                        assert total_remaining_res_time >= - \
                            0.001, "Remaining res time: {}".format(
                                total_remaining_res_time)

        return remaining_time

    def get_current_accuracy(self):
        return self.acc

    def get_completion_accuracy(self):
        return self.train_fn(
            self.init_train_duration + self.total_train_duration)

    def __str__(self):
        return self.name


class InferenceJob(EkyaJob):
    """The class which simulates inference job."""

    def __init__(self, name, accuracy, model_name, subsampling,
                 inference_camera_profile, resource_alloc):
        """Initialize an inference job.

        Args
            name(str): name of the inference job.
            accuracy(float): accuracy of inference job [0, 1.0].
            model_name(str): model name. e.g. resnet18.
            subsampling(pandas series): a list of subsampling rate.
            inference_camera_profile(pandas series): inference profile with
                respect to subsampling.
            resource_alloc(float): gpu resource allocated.
        """
        self.start_acc = accuracy
        self.acc = self.start_acc
        self.job_name = name
        self.model_name = model_name
        self.subsampling = subsampling
        self.inference_camera_profile = inference_camera_profile

        # Multiplication factor 0 to 1
        self.perf_vs_res_func = get_infer_profile(
            self.subsampling, self.inference_camera_profile,
            max_inference_resources=INFER_MAX_RES_DICT[self.model_name])[0]
        super(InferenceJob, self).__init__(self.job_name, resource_alloc)
        self.res_acc_scale = self.perf_vs_res_func(self.current_resource_alloc)

    def set_resource_alloc(self, resource_alloc):
        """Update gpu resource allocated to the job."""
        if resource_alloc == self.current_resource_alloc:
            return
        self.current_resource_alloc = resource_alloc
        self.res_acc_scale = self.perf_vs_res_func(self.current_resource_alloc)

    def get_accuracy(self):
        # return self.acc * self.perf_vs_res_func(self.current_resource_alloc)
        return self.acc * self.res_acc_scale

    def set_base_accuracy(self, acc):
        self.acc = acc

    def update_model_name(self, model_name):
        """Update the model name used in the job.

        Update the model name and also update the inference profile.

        Args
            model_name(str): model name e.g. resnet18.
        """
        self.model_name = model_name
        self.perf_vs_res_func = get_infer_profile(
            self.subsampling, self.inference_camera_profile,
            max_inference_resources=INFER_MAX_RES_DICT[self.model_name])[0]

    def step(self, wall_time):
        return self.get_accuracy()

    # def reset(self):
    #     self.acc = self.start_acc
    #     super(InferenceJob, self).__init__(self.job_name, resource_alloc=0)
