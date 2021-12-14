import logging
import copy
from collections import defaultdict

import matplotlib.pyplot as plt
from numpy import arctanh, tanh
import numpy as np
from numpy.testing import assert_almost_equal

INFINITY = 9999999

class EkyaJob(object):
    def __init__(self, name, resource_alloc):
        self.name = name
        self.current_resource_alloc = resource_alloc
        self.done = False

    def set_resource_alloc(self, resource_alloc):
        self.current_resource_alloc = resource_alloc

    def is_done(self):
        return self.done

    def __str__(self):
        return self.name


class TrainingJob(EkyaJob):
    def __init__(self, name, train_acc_vs_t_function, init_train_duration, total_train_duration, resource_alloc, inference_job=None):
        self.train_fn = train_acc_vs_t_function # GPU Cycles
        self.init_train_duration = init_train_duration
        self.trained_duration = self.init_train_duration
        self.acc = self.train_fn(self.trained_duration)
        self.inference_job = inference_job
        self.total_train_duration = total_train_duration
        self.job_name = name
        super(TrainingJob, self).__init__(self.job_name, resource_alloc)

    # def reset(self):
    #     self.trained_duration = self.init_train_duration
    #     self.acc = self.train_fn(self.trained_duration)
    #     super(TrainingJob, self).__init__(self.job_name, resource_alloc=0)

    def run_train(self, wall_time):
        res_time = wall_time * self.current_resource_alloc
        self.trained_duration += res_time
        self.acc = self.train_fn(self.trained_duration + res_time)
        if self.inference_job:
            self.inference_job.set_base_accuracy(self.acc)
        if self.total_train_duration <= self.trained_duration:
            self.done = True
        return self.acc

    def step(self, wall_time):
        if not self.is_done():
            old_acc = self.acc
            self.run_train(wall_time)
            delta_acc = self.acc - old_acc
        elif (self.current_resource_alloc != 0):
            print("Warning: Job {} done but still allocated resources {} and being stepped. Doing nothing..".format(self.name, self.current_resource_alloc))
        return self.acc

    def completion_time(self, allocation_over_walltime):
        # allocation_over_walltime = {time: alloc} eg. {0: 0.25, 10: 1}
        # This method answers "If the job is given allocation_over_time resources for the remaining duration, how much time will it take to complete?"
        total_remaining_res_time = (self.total_train_duration - self.init_train_duration)
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
                        # The job will complete in this allocation block. Compute the total wall time
                        remaining_time += total_remaining_res_time / allocation
                        break
                    else:
                        # We'll need to run for this allocation and see if it completes in the next allocation
                        total_remaining_res_time -= block_duration * allocation
                        assert total_remaining_res_time >= 0

        return remaining_time

    def get_current_accuracy(self):
        return self.acc


class InferenceJob(EkyaJob):
    def __init__(self, name, accuracy, perf_vs_resource_function, resource_alloc):
        self.start_acc = accuracy
        self.acc = self.start_acc
        self.job_name = name
        self.perf_vs_resource_function = perf_vs_resource_function  # Multiplication factor 0 to 1
        super(InferenceJob, self).__init__(self.job_name, resource_alloc)

    def get_accuracy(self):
        return self.acc * self.perf_vs_resource_function(self.current_resource_alloc)

    def set_base_accuracy(self, acc):
        self.acc = acc

    def step(self, wall_time):
        return self.get_accuracy()

    # def reset(self):
    #     self.acc = self.start_acc
    #     super(InferenceJob, self).__init__(self.job_name, resource_alloc=0)

def slowed_acc(acc, contention_slowdown=0.9):
    return acc * contention_slowdown


def optimus_fn(t, T):
    if t == 0:
        return 0
    else:
        return 1 / ((1 / (t * 10 * 1 / T)) + 1) * 100  # Hits 0.9 acc at time T. *100 for accuracy.


def inv_optimus_fn(a, T):  # Maps accuract to time
    if a == 0:
        return 0
    else:
        return a / (10 / T * (100 - a))


def linear_fn(t, k):
    return k * t


def inv_linear_fn(a, k):
    return a / k


def tanh_fn(t, scale_factor, shift=0):
    return (tanh(((t - shift) * 2 * scale_factor - scale_factor)) + 1) / 2


def inv_tanh_fn(p, k):
    (k - arctanh(1 - 2 * p)) / 2 * k


def get_linear_fn(k):
    return lambda t: linear_fn(t, k), lambda a: inv_linear_fn(a, k)


def get_optimus_fn(T):
    return lambda t: optimus_fn(t, T), lambda a: inv_optimus_fn(a, T)


def get_tanh_fn(k):
    return lambda t: tanh_fn(t, k), lambda a: inv_tanh_fn(a, k)

def thief_scheduler(training_jobs, inference_jobs, total_resources, **kwargs):
    # Start off as a fair allocator, then "steal" resources from other jobs till you max out the accuracy.
    # This may potentially require multiple nested simulators to evaluate each configuration.
    # Return allocation for the entire retraining period as a dict with {timestamp: allocation}.

    def create_sim_env(training_jobs, inference_jobs, start_allocation):
        new_training_jobs = []
        new_inference_jobs = []
        new_allocation = {}
        retraining_period = kwargs['retraining_period']

        # Deep copy jobs
        for jobname, allocation in start_allocation.items():
            if jobname in training_jobs:
                job = training_jobs[jobname]
            else:
                job = inference_jobs[jobname]
            new_job = copy.deepcopy(job)
            new_allocation[new_job.name] = allocation
            if isinstance(job, InferenceJob):
                new_inference_jobs.append(new_job)
            elif isinstance(job, TrainingJob):
                new_training_jobs.append(new_job)
            else:
                raise Exception("Got unsupported job type {}.".format(new_job))
            assert new_job != job

        temp_sim = simulator(new_training_jobs, new_inference_jobs, total_resources, quantum_size=1, retraining_period=retraining_period,
                             scheduling_algo=fair_scheduler, initial_allocation=new_allocation, verbose=False)
        return temp_sim


    all_jobs = list(training_jobs.values()) + list(inference_jobs.values())
    # Initial fair allocation:
    fair_alloc = {j.name: total_resources/len(all_jobs) for j in all_jobs}

    best_period_allocation = {0: fair_alloc}

    # print("Init alloc: {}".format(period_allocation))

    # Iterate over all jobs to steal resources
    allocationid_to_score_map = {}
    allocationid_map = {}   # Maps all allocations to a hashable id
    alloc_id = 0

    for candidate_job in all_jobs:
        for victim_job in [x for x in all_jobs if x != candidate_job]:
            should_steal = True
            max_steal_resources = best_period_allocation[0][victim_job.name]
            for steal_amount in np.arange(0, max_steal_resources, 0.1):
                if should_steal:
                    temp_period_allocation = copy.deepcopy(best_period_allocation)
                    temp_curr_allocation = temp_period_allocation[0]
                    new_victim_handle = victim_job.name
                    new_candidate_handle = candidate_job.name
                    # new_victim_handle = next((x for x in temp_curr_allocation.keys() if x.name == victim_job.name), None)
                    # new_candidate_handle = next((x for x in temp_curr_allocation.keys() if x.name == candidate_job.name), None)
                    victim_resources = temp_curr_allocation[new_victim_handle]

                    # Cap the steal to available resources:
                    if victim_resources < steal_amount:
                        stolen_resources = victim_resources
                        should_steal = False
                    else:
                        stolen_resources = steal_amount

                    # Commit the steal
                    temp_curr_allocation[new_victim_handle] -= stolen_resources
                    temp_curr_allocation[new_candidate_handle] += stolen_resources

                    # print("Temp curr alloc: {}".format(temp_curr_allocation))

                    # New temp_allocation is ready, evaluate it's goodness with a simulator.
                    temp_sim = create_sim_env(training_jobs, inference_jobs, temp_curr_allocation) #TODO: is this bad? The simulator actually creates the best schedule.
                    result = temp_sim.step_till_completion()

                    allocationid_map[alloc_id] = temp_sim.period_allocation
                    allocationid_to_score_map[alloc_id] = result['meta']['inf_mean_auc']
                    alloc_id+=1
            # TODO: Uncomment this.
            #best_period_allocation = {0: allocationid_map[max(allocationid_to_score_map, key=allocationid_to_score_map.get)][0]}

    # Pick the allocation with max score
    print(allocationid_map)
    print(allocationid_to_score_map)
    best_allocation = allocationid_map[max(allocationid_to_score_map, key=allocationid_to_score_map.get)]
    return best_allocation


def fair_scheduler(training_jobs, inference_jobs, total_resources, **kwargs):
    # allocate each job equal amount of resources.
    # Return allocation for the entire retraining period as a dict with {timestamp: allocation}.
    all_jobs = list(training_jobs.values()) + list(inference_jobs.values())
    initial_allocation = kwargs.get('initial_allocation', {})
    if not initial_allocation:
        initial_allocation = {j.name: total_resources/len(all_jobs) for j in all_jobs}    # Fair sharing
    allocation = {0: initial_allocation}


    # Perform reallocation when jobs complete

    period_allocation = fair_reallocator(training_jobs, inference_jobs, allocation)

    # Sanity check:
    for t, instant_alloc in period_allocation.items():
        assert_almost_equal(sum(instant_alloc.values()), total_resources, err_msg="Got total resources: {}".format(
            sum(instant_alloc.values())))
    return period_allocation


def fair_reallocator(training_jobs, inference_jobs, initial_allocation):
    period_allocation = copy.copy(initial_allocation)
    running_training_jobs = list(training_jobs.values())  # Shallow copy
    while running_training_jobs:
        train_completion_times = {}  # Job: time
        for j in running_training_jobs:
            job_allocation = {t: alloc[j.name] for t, alloc in period_allocation.items()}
            train_completion_times[j] = j.completion_time(job_allocation)
        # print(train_completion_times)
        completed_job, completion_time = min(train_completion_times.items(), key=lambda x: x[1])
        # print("Job {} completes at {}".format(completed_job, completion_time))

        last_allocation_time, last_instant_alloc = max(period_allocation.items(), key=lambda x: x[0])

        # Get resources of killed job and fairly redistribute
        running_training_jobs.remove(completed_job)
        current_jobs = running_training_jobs + list(inference_jobs.values())#.remove(completed_job)
        relinquished_resources = last_instant_alloc[completed_job.name]
        new_alloc_per_job = relinquished_resources / len(current_jobs)
        new_instant_alloc = {}
        for j in (list(set(training_jobs.values()) - set(running_training_jobs))):
            # Set allocation to zero for completed training jobs
            new_instant_alloc[j.name] = 0
        for j in current_jobs:
            new_instant_alloc[j.name] = last_instant_alloc[j.name] + new_alloc_per_job  # Redistribute

        period_allocation[completion_time] = new_instant_alloc
    return period_allocation


class simulator(object):
    def __init__(self, training_jobs, inference_jobs, total_resources, quantum_size=1, retraining_period=100,
                 scheduling_algo=fair_scheduler, initial_allocation=None, verbose=True, sim_name='unnamed'):
        self.training_jobs = {j.name: j for j in training_jobs}
        self.inference_jobs = {j.name: j for j in inference_jobs}
        self.total_resources = total_resources

        self.scheduling_algo = scheduling_algo

        self.current_t = 0
        self.quantum_size = quantum_size
        self.instantaneous_allocation = {}
        self.period_allocation = {}
        self.initial_allocation = initial_allocation
        self.retraining_period = retraining_period

        self.logger = logging.getLogger(sim_name)
        self.logger.handlers = [logging.StreamHandler()]
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARN)

        self.compute_schedule()

        self.metrics = defaultdict(list)

    def compute_schedule(self):
        custom_args = {'initial_allocation': self.initial_allocation,
                       'retraining_period': self.retraining_period}
        self.period_allocation = self.scheduling_algo(self.training_jobs, self.inference_jobs, self.total_resources, **custom_args)
        self.logger.info("Period allocation: {}".format(self.period_allocation))

    def compute_retrain_period_accuracy(self, tentative_period_allocation):
        pass

    def update_instantaneous_allocation(self, allocation):
        for k in sorted(allocation.keys()):
            if self.current_t >= k:
                self.instantaneous_allocation = allocation[k]
            else:
                break

    def analyze_metrics(self):
        # Compute mean inference AUC and add to metrics dict
        self.metrics['meta'] = {}
        means = []
        for j, result in self.metrics.items():
            if isinstance(j, InferenceJob):
                mean_accuracy = sum(r[1] for r in result) / len(result)
                means.append(mean_accuracy)
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
            self.logger.info(self.current_t)
            return False

    def step_jobs(self):
        for jobname, allocation in self.instantaneous_allocation.items():
            if jobname in self.training_jobs:
                job = self.training_jobs[jobname]
            else:
                job = self.inference_jobs[jobname]
            job.set_resource_alloc(allocation)
            if isinstance(job, TrainingJob):
                self.metrics[job].append([self.current_t, job.step(self.quantum_size)])
            if isinstance(job, InferenceJob):
                inf_accuracy = job.step(self.quantum_size)
                self.metrics[job].append([self.current_t, inf_accuracy])
            if job.is_done():
                self.logger.info("Job {} is done.".format(job))

    def step_till_completion(self):
        done = False
        while not done:
            done = self.step()
        return self.metrics


if __name__ == '__main__':
    a_conv_time = 30
    b_conv_time = 50
    target_start_accuracy = 50
    a_func, a_inv_func = get_optimus_fn(a_conv_time)
    b_func, b_inv_func = get_optimus_fn(b_conv_time)
    init_time_a = a_inv_func(target_start_accuracy)
    init_time_b = b_inv_func(target_start_accuracy)
    A_inference = InferenceJob("A_Inference", target_start_accuracy, get_linear_fn(1)[0], resource_alloc=0)
    B_inference = InferenceJob("B_Inference", target_start_accuracy, get_linear_fn(1)[0], resource_alloc=0)
    A_train = TrainingJob("A_Train", a_func, init_time_a, a_conv_time, resource_alloc=0, inference_job=A_inference)
    B_train = TrainingJob("B_Train", b_func, init_time_b, b_conv_time, resource_alloc=0, inference_job=B_inference)

    training_jobs = [A_train, B_train]
    inference_jobs = [A_inference, B_inference]

    sim = simulator(training_jobs, inference_jobs, 1, retraining_period=200)
    print(sim.step_till_completion())