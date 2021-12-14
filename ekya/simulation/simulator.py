"""This module defines the simulators used in Ekya simulation."""
import copy
import logging
import time
from collections import defaultdict

from ekya.simulation.jobs import InferenceJob, TrainingJob
from ekya.simulation.constants import OPT


class simulator(object):
    """A simulator class which runs for a single retraining window."""

    def __init__(self, training_jobs, inference_jobs, total_resources,
                 scheduling_algo, quantum_size=1, retraining_period=100,
                 initial_allocation=None, period_allocation=None, verbose=True,
                 sim_name='unnamed', golden_model_delay=0):
        """Initialize simulator object.

        Args
            training_jobs(list): a list TrainingJobs to be simulated.
            inference_jobs(list): a list of InferenceJob to be simulated.
            total_resources(float): total number of GPU resources
            scheduling_algo(function): a scheduler to be used in simulation.
            quantum_size(int): the minimum step length in the simulator in
                seconds.
            retraining_period(int): the length of a retraining window in
                seconds.
            initial_allocation(dict)
            period_allocation(dict)
            verbose(bool): Print out logs if True. Otherwise, no printing.
            sim_name(str): the name of the simulator.
        """
        self.training_jobs = {j.name: j for j in training_jobs}
        self.inference_jobs = {j.name: j for j in inference_jobs}
        self.total_resources = total_resources

        self.scheduling_algo = scheduling_algo

        self.current_t = 0
        self.quantum_size = quantum_size

        # initialize the allocation to even distribution of gpu resources over
        # all inference jobs
        self.instantaneous_allocation = {
            j.name: total_resources / len(inference_jobs)
            for j in inference_jobs}  # Fair sharing
        if not period_allocation:
            self.period_allocation = {}
        else:
            self.period_allocation = period_allocation
        self.initial_allocation = initial_allocation
        self.retraining_period = retraining_period

        self.logger = logging.getLogger(sim_name)
        self.logger.handlers = [logging.StreamHandler()]
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARN)

        self.golden_model_delay = golden_model_delay

        if not self.period_allocation:
            self.compute_schedule()

        self.metrics = defaultdict(list)

    def compute_schedule(self):
        self.logger.info("No period allocation found, running scheduler.")
        custom_args = {'initial_allocation': self.initial_allocation,
                       'retraining_period': self.retraining_period - self.golden_model_delay}
        self.period_allocation = self.scheduling_algo(
            self.training_jobs, self.inference_jobs, self.total_resources,
            **custom_args)
        self.logger.info("Period allocation: {}".format(
            self.period_allocation))

    def update_instantaneous_allocation(self, allocation):
        for k in sorted(allocation.keys()):
            if self.current_t >= k + self.golden_model_delay:
                self.instantaneous_allocation = allocation[k]
            else:
                break

    def update_instantaneous_allocation_optimized(self, allocation):
        """"""
        for k in sorted(allocation.keys()):
            if self.current_t >= k + self.golden_model_delay:
                self.instantaneous_allocation = allocation[k]
            else:
                break

        # compute the step size
        if self.current_t == self.retraining_period:
            return 1
        elif self.current_t == int(k + self.golden_model_delay):
            return 1
        elif self.current_t < k + self.golden_model_delay:
            return int(min(k + self.golden_model_delay,
                           self.retraining_period) - self.current_t)
        else:
            return self.retraining_period - self.current_t

    def analyze_metrics(self):
        """Compute mean inference AUC and add to metrics dict."""
        self.metrics['meta'] = {}
        means = []
        for j, result in self.metrics.items():
            if isinstance(j, InferenceJob):
                mean_accuracy = sum(r[1] for r in result) / len(result)
                means.append(mean_accuracy)
        self.metrics['meta']['inf_mean_auc'] = sum(means) / len(means)

    def analyze_metrics_optimized(self):
        # Compute mean inference AUC and add to metrics dict
        self.metrics['meta'] = {}
        means = []
        for j, result in self.metrics.items():
            if isinstance(j, InferenceJob):
                # compute the average inference accuracy over the entire
                # retraining period
                mean_accuracy = 0
                i = 1
                for k in range(int(self.retraining_period+1)):
                    if i < len(result) and k >= result[i][0]:
                        i += 1
                    mean_accuracy += result[i-1][1]
                mean_accuracy = mean_accuracy / (self.retraining_period+1)
                means.append(mean_accuracy)
        # avearge over all mean accuracy of inference jobs
        self.metrics['meta']['inf_mean_auc'] = sum(means) / len(means)

    def step(self):
        if self.current_t > self.retraining_period:
            self.analyze_metrics()
            self.logger.info("Simulator steps done.")
            return True
        else:
            self.update_instantaneous_allocation(self.period_allocation)
            self.step_jobs()
            self.current_t += self.quantum_size
            if self.current_t % 50 == 0:
                self.logger.info(self.current_t)
            return False

    def step_optimized(self):
        if self.current_t > self.retraining_period:
            self.analyze_metrics_optimized()
            self.logger.info("Simulator steps done.")
            return True
        else:
            step_len = self.update_instantaneous_allocation_optimized(
                self.period_allocation)
            self.step_jobs_optimized(step_len)
            self.current_t += step_len
            if self.current_t % 50 == 0:
                self.logger.info(self.current_t)
            return False

    def step_jobs(self):
        for jobname, allocation in self.instantaneous_allocation.items():
            if jobname in self.inference_jobs:
                job = self.inference_jobs[jobname]
                job.set_resource_alloc(allocation)
                if isinstance(job, InferenceJob):
                    inf_acc = job.get_accuracy()
                    self.metrics[job].append(
                        [self.current_t, inf_acc, allocation])
                    job.step(self.quantum_size)
                if job.is_done():
                    self.logger.info("Job {} is done.".format(job))
        for jobname, allocation in self.instantaneous_allocation.items():
            if jobname in self.training_jobs:
                job = self.training_jobs[jobname]
                job.set_resource_alloc(allocation)
                if isinstance(job, TrainingJob):
                    self.metrics[job].append(
                        [self.current_t, job.get_accuracy(), allocation])
                    job.step(self.quantum_size)
                if job.is_done():
                    self.logger.info("Job {} is done.".format(job))

    def step_jobs_optimized(self, step_len):
        for jobname, allocation in self.instantaneous_allocation.items():
            if jobname in self.inference_jobs:
                job = self.inference_jobs[jobname]
                job.set_resource_alloc(allocation)
                # if isinstance(job, InferenceJob):
                self.metrics[job].append(
                    [self.current_t, job.get_accuracy(), allocation])
                job.step(1)
                # if job.is_done():
                #     self.logger.info("Job {} is done.".format(job))

        for jobname, allocation in self.instantaneous_allocation.items():
            if jobname in self.training_jobs:
                job = self.training_jobs[jobname]
                job.set_resource_alloc(allocation)
                # if isinstance(job, TrainingJob):
                self.metrics[job].append(
                    [self.current_t, job.get_accuracy(), allocation])
                job.step(1)
                # if job.is_done():
                #     self.logger.info("Job {} is done.".format(job))

        if step_len - 1 > 0:
            for jobname, allocation in self.instantaneous_allocation.items():
                if jobname in self.inference_jobs:
                    job = self.inference_jobs[jobname]
                    # job.set_resource_alloc(allocation)
                    # if isinstance(job, InferenceJob):
                    job.step(step_len-1)
                    # if job.is_done():
                    #     self.logger.info("Job {} is done.".format(job))

            for jobname, allocation in self.instantaneous_allocation.items():
                if jobname in self.training_jobs:
                    job = self.training_jobs[jobname]
                    # job.set_resource_alloc(allocation)
                    # if isinstance(job, TrainingJob):
                    job.step(step_len-1)
                    # if job.is_done():
                    #     self.logger.info("Job {} is done.".format(job))

    def step_till_completion(self):
        done = False
        while not done:
            if OPT:
                done = self.step_optimized()
            else:
                done = self.step()
        return self.metrics


class MultiPeriodSimulator(object):
    """A simulator which runs for multiple retraining windows.

    Reuses the single period simulator and runs it multiple times, but updates
    job states across retraining windows.
    """

    def __init__(self, cameras, total_resources, scheduler, retraining_period,
                 task_ids=["1"], incomplete_jobs_checkpoint=False,
                 golden_model_delay=0):
        """Initialize MultiPeriodSimulator object.

        Args
            cameras(list)
            total_resources(float): total number of GPU resources
            scheduler(function): a scheduler to be used in simulation.
            retraining_period(int): the length of a retraining window in
                seconds.
            verbose(bool): Print out logs if True. Otherwise, no printing.
            task_ids(list): list of tadk id used in the simulation.
            sim_name(str): the name of the simulator.
        """
        self.incomplete_jobs_checkpoint = incomplete_jobs_checkpoint
        self.retraining_period = retraining_period
        self.cameras = cameras
        self.total_resources = total_resources
        self.scheduler = scheduler
        self.task_ids = task_ids
        self.golden_model_delay = golden_model_delay

    def get_job_cfgs(self, task_id):
        job_pairs = []
        for camera in self.cameras:
            camera.generate_training_configurations(
                task_id)   # Generate new training configs
            camera.generate_oracle_configurations(
                task_id)   # Generate new training configs
            train_configs = camera.get_training_configurations()
            oracle_configs = camera.get_oracle_configurations()
            job_pairs.append((camera.get_inference_job(),
                              train_configs, oracle_configs))
        return job_pairs

    @staticmethod
    def get_job_instance(job_pairs, jobname):
        for inf_job, training_configs in job_pairs:
            if inf_job.name == jobname:
                return inf_job
            else:
                for train_cfg in training_configs:
                    if train_cfg.name == jobname:
                        return train_cfg
        raise Exception("Job {} not found.".format(jobname))

    def step_till_completion(self):
        """Step the simulator to the end of all tasks.

        Return
            results(dict): the accuracy of all training and inference jobs.
            period_allocation_log(dict): period allocation of all tasks.
            scheduler_t_used(dict): scheduler time usage of all tasks.
        """
        results = {}
        period_allocation_log = {}  # store the schdule in each task
        scheduler_t_used = {}  # store the scheduler's time usage in each task
        for task_id in self.task_ids:
            print("Running task {}".format(task_id))

            # Update all cameras to reset their inference accuracy to the one
            # mentioned in profile. This is assuming models restart from
            # scratch
            for camera in self.cameras:
                camera.reset_current_accuracy(task_id)

            # Get training profiles here
            job_cfgs = self.get_job_cfgs(task_id)
            # Get new period allocation here given inference job and training
            # profiles from Camera object
            sched_job_pairs = [(jc[0], jc[1]) for jc in job_cfgs]  # Training
            oracle_job_pairs = [(jc[0], jc[2]) for jc in job_cfgs]    # Oracle

            start_t = time.time()
            period_allocation = self.scheduler(
                sched_job_pairs, self.total_resources,
                self.retraining_period - self.golden_model_delay)
            t_used = time.time() - start_t
            print(f'cost {t_used} seconds')
            scheduler_t_used[task_id] = t_used
            period_allocation_log[task_id] = period_allocation

            training_jobs = []
            inference_jobs = []

            for jobname in period_allocation[0].keys():
                job = self.get_job_instance(oracle_job_pairs, jobname)
                if isinstance(job, InferenceJob):
                    inference_jobs.append(job)
                elif isinstance(job, TrainingJob):
                    training_jobs.append(job)
                else:
                    raise Exception
            sim = simulator(training_jobs, inference_jobs,
                            self.total_resources, None,
                            retraining_period=self.retraining_period,
                            sim_name='period{}'.format(task_id),
                            period_allocation=period_allocation,
                            golden_model_delay=self.golden_model_delay)
            results[task_id] = sim.step_till_completion()

            # Get metrics from all inference jobs and update the configurations
            # in cameras and global result
            # Use training jobs to update accuracy or use inference jobs to
            # update? Key question - do incomplete training jobs checkpoint at
            # the end of retraining period?
            # for train_job in training_jobs:
            #     acc = train_job.get_current_accuracy()
            #     if train_job.is_done() or self.incomplete_jobs_checkpoint:
            #         camera = train_to_camera_map[train_job.name]
            #         camera.set_current_accuracy(acc)
            #     else:
            #         # Do not checkpoint if training job did not complete or
            #         pass
        results['meta'] = {'overall_inf_mean_auc': sum(
            [results[r]['meta']['inf_mean_auc'] for r in results]) /
            len(results)}
        return results, period_allocation_log, scheduler_t_used


def create_sim_env(training_jobs, inference_jobs, start_allocation,
                   retraining_period, total_resources, scheduler):
    """Create a simulator.

    Args
        training_jobs(dict): a dict mapping training job names to training jobs
        inference_jobs(dict): a dict mapping inference job names to inference
            jobs.
    """
    new_training_jobs = []
    new_inference_jobs = []
    new_allocation = {}

    # Deep copy jobs
    for jobname, allocation in start_allocation.items():
        if jobname in training_jobs:
            job = training_jobs[jobname]
        else:
            job = inference_jobs[jobname]
        # new_job = copy.deepcopy(job)
        new_job = copy.copy(job)

        new_allocation[new_job.name] = allocation
        if isinstance(job, InferenceJob):
            new_inference_jobs.append(new_job)
        elif isinstance(job, TrainingJob):
            new_training_jobs.append(new_job)
        else:
            raise Exception("Got unsupported job type {}.".format(new_job))
        assert new_job != job

    # Relink deepcopies of training jobs to their inference jobs:
    for job in new_training_jobs:
        if job.inference_job:
            new_inference_job = next(
                x for x in new_inference_jobs
                if x.name == job.inference_job.name)
            job.inference_job = new_inference_job
        elif job.inference_job is None:
            print("Warning: No inference job associated for {}, not linking deep "
                  "copy.".format(job.name))

    temp_sim = simulator(new_training_jobs, new_inference_jobs,
                         total_resources, quantum_size=1,
                         retraining_period=retraining_period,
                         scheduling_algo=scheduler,
                         initial_allocation=new_allocation, verbose=False)
    return temp_sim
