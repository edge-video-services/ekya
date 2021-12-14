import logging
import copy
from collections import defaultdict

import pandas as pd
from numpy import arctanh, tanh
import numpy as np
from numpy.testing import assert_almost_equal

INFINITY = 9999999
INSTA_CHECKPOINT = False


class ReversibleDictionary(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rev = {id(v): k for k, v in self.items()}

    def __delitem__(self, k):
        del self.rev[id(self[k])]
        del super()[k]

    def __setitem__(self, k, v):
        try:
            del self.rev[id(self[k])]
        except KeyError:
            pass
        super()[k] = v
        self.rev[id(v)] = k

    def lookup(self, v):
        return self.rev[id(v)]

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
            if INSTA_CHECKPOINT:
                self.inference_job.set_base_accuracy(self.acc)
        if self.total_train_duration <= self.trained_duration:
            self.done = True
            if not INSTA_CHECKPOINT and self.inference_job:
                self.inference_job.set_base_accuracy(self.acc)
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

    def get_completion_accuracy(self):
        return self.train_fn(self.init_train_duration + self.total_train_duration)

    def __str__(self):
        return self.name


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

def profile_fn(t, profile_time, profile_acc):
    # Returns a continuous function from the given profile time and acc values.
    return np.interp(t, profile_time, profile_acc)

def optimus_fn(t, T, K = 1):
    if t == 0:
        return 0
    else:
        return 1 / ((1 / (t * 10 * 1 / T)) + 1) * 100 * K  # Hits 0.9 acc at time T. *100 for accuracy.


def inv_optimus_fn(a, T, K = 1):  # Maps accuracy to time
    if a == 0:
        return 0
    else:
        return a / ((10 / T) * (100*K - a))

def k_optimus_fn(a, T, t):
    # Returns the value of k for a given a, T and t
    # solve a=1 / ((1 / (t * 10 * 1 / (t+d))) + 1) * 100 * K, b=1 / ((1 / ((t+d) * 10 * 1 / (t+d))) + 1) * 100 * K for K, t
    return ((a*T) / 10*t + a) / 100

def linear_fn(t, k):
    return min(k * t, 1)


def inv_linear_fn(a, k):
    return a / k


def tanh_fn(t, scale_factor, shift=0):
    return (tanh(((t - shift) * 2 * scale_factor - scale_factor)) + 1) / 2


def inv_tanh_fn(p, k):
    (k - arctanh(1 - 2 * p)) / 2 * k


def get_linear_fn(k):
    return lambda t: linear_fn(t, k), lambda a: inv_linear_fn(a, k)


def get_optimus_fn(T, K = 1):
    return lambda t: optimus_fn(t, T, K), lambda a: inv_optimus_fn(a, T, K)

def get_tanh_fn(k):
    return lambda t: tanh_fn(t, k), lambda a: inv_tanh_fn(a, k)

def get_infer_profile(max_inference_resources = 1,
                      profile_path = 'real_inference_profiles.csv',
                      camera = 'c1'):
    data = pd.read_csv(profile_path)
    # effective_subsample_rate is max_inference_resources/res + 0.0000001 to avoid divide by zero.
    return lambda res: np.interp(max_inference_resources/(res+0.000001), data['subsampling'], data[camera], right=0), lambda acc_scale: NotImplementedError# max_inference_resources/np.interp(acc_scale, data[camera], data['subsampling'])

def thief_scheduler(training_jobs, inference_jobs, total_resources, steal_increment = 0.1, **kwargs):
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

        # Relink deepcopies of training jobs to their inference jobs:
        for job in new_training_jobs:
            if job.inference_job:
                new_inference_job = next(x for x in new_inference_jobs if x.name==job.inference_job.name)
                job.inference_job = new_inference_job
            elif job.inference_job == None:
                print("Warning: No inference job associated, not linking deep copy.")

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
            for steal_amount in (list(np.arange(0, max_steal_resources, steal_increment)) + [max_steal_resources]):
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
            best_period_allocation = {0: allocationid_map[max(allocationid_to_score_map, key=allocationid_to_score_map.get)][0]}

    # Pick the allocation with max score
    # print(allocationid_map)
    # print(allocationid_to_score_map)
    debug = [allocationid_map, allocationid_to_score_map]
    best_allocation = allocationid_map[max(allocationid_to_score_map, key=allocationid_to_score_map.get)]
    return best_allocation, debug


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

def single_camera_optimizer(training_jobs, inference_job, total_resources, retraining_period,
                            training_res_alloc = 0.5, inference_res_alloc = 0.5):
    '''

    :param training_jobs: The training configurations to pick from
    :param inference_job: The inference job
    :param total_resources: Total amount of resources
    :return: best training config and debug info
    '''
    # Iterate over all configurations to find best one
    debug = None
    if len(training_jobs) == 0:
        return None, debug
    allocationid_to_score_map = {}
    allocationid_map = {}  # Maps all allocations to a hashable id
    alloc_id = 0
    for training_job in training_jobs:
        candidate_training_job = copy.deepcopy(training_job)
        candidate_inference_job = copy.deepcopy(inference_job)
        candidate_inference_job.name = "inference_job"
        candidate_training_job.inference_job = candidate_inference_job

        initial_allocation = {0: {candidate_inference_job.name: inference_res_alloc,
                                  candidate_training_job.name: training_res_alloc}}
        period_allocation = fair_reallocator({candidate_training_job.name: candidate_training_job},
                                             {candidate_inference_job.name: candidate_inference_job},
                                             initial_allocation)

        temp_sim = simulator([candidate_training_job], [candidate_inference_job], total_resources, quantum_size=1,
                             retraining_period=retraining_period, period_allocation=period_allocation, verbose=False)
        result = temp_sim.step_till_completion()

        allocationid_map[alloc_id] = period_allocation
        allocationid_to_score_map[alloc_id] = result['meta']['inf_mean_auc']
        alloc_id+=1

    best_allocation_jobnames = allocationid_map[max(allocationid_to_score_map, key=allocationid_to_score_map.get)][0].keys()
    training_job_name = next(x for x in best_allocation_jobnames if x != "inference_job")
    debug = [allocationid_to_score_map]
    return next(x for x in training_jobs if x.name == training_job_name), debug

def single_camera_naive(training_configs, retraining_period, training_res_alloc, allow_no_train=True):
    '''
    A slightly dumb variant of Single Camera Optimizer. Picks a training configuration which fits in the retraining window
    and gives the max accuracy, even though does not leave enough time to exploit it.
    :param training_configs: The training configurations to pick from
    :return: training config with max accuracy that fits in the training window
    '''
    # Sort jobs by their completion accuracy and iterate over all
    # configurations to find with max accuracy that fits in retraining_period
    sorted_training_configs = sorted(training_configs, key=lambda x: x.get_completion_accuracy(), reverse=True)
    candidate_job = None
    for training_job in sorted_training_configs:
        trg_job_allocation = {0: training_res_alloc}
        completion_time = training_job.completion_time(trg_job_allocation)
        if completion_time < retraining_period:
            candidate_job = training_job
            break

    if candidate_job is None:
        if allow_no_train:
            print("No job found to fit in the retraining window. Not retraining this period.")
            return None
        else:
            raise Exception("No job found to fit in the retraining window.")

    return candidate_job


def single_camera_dumb(training_configs):
    '''
    A dumb cloud-like scheduler which picks the config with max accuracy, irrespective of how much time it takes.
    :param training_configs: The training configurations to pick from
    :return: training config with max accuracy that fits in the training window
    '''
    # Sort jobs by their completion accuracy and pick the first one.
    sorted_training_configs = sorted(training_configs, key=lambda x: x.get_completion_accuracy(), reverse=True)
    candidate_job = sorted_training_configs[0]
    return candidate_job

def thief_sco_scheduler(job_pairs, total_resources, retraining_period, iterations = 10, steal_increment=0.1):
    '''
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return:
    '''
    inference_trg_map = ReversibleDictionary({inference_job: training_configs
                        for inference_job, training_configs in job_pairs})
    current_best_config = {}

    # Set initial fair allocations
    per_job_alloc = total_resources/(len(inference_trg_map.keys())*2)
    res_alloc = {k: [per_job_alloc, per_job_alloc] for k in inference_trg_map.keys()}   # [inference, training] resources.

    for i in range(iterations):
        print("Iter {}, alloc: {}".format(i, res_alloc))
        # First do SCO
        for inference_job, training_configs in inference_trg_map.items():
            best_trg_config_name, _ = single_camera_optimizer(training_configs, inference_job,
                                                           total_resources=total_resources,
                                                           retraining_period=retraining_period,
                                                           inference_res_alloc=res_alloc[inference_job][0],
                                                           training_res_alloc=res_alloc[inference_job][1]
                                                           )
            if best_trg_config_name:
                best_trg_config = next(x for x in training_configs if x.name == best_trg_config_name.name)
            else:
                best_trg_config = None
            current_best_config[inference_job] = best_trg_config
            print("Best config for {}: {}".format(inference_job.name, best_trg_config))
        training_jobs = list([x for x in current_best_config.values() if x!=None])
        inference_jobs = list(current_best_config.keys())

        # Then run thief scheduler
        custom_args = {'retraining_period': retraining_period}
        period_allocation, _ = thief_scheduler({t.name: t for t in training_jobs},
                                            {t.name: t for t in inference_jobs},
                                            total_resources,
                                            steal_increment = steal_increment,
                                            **custom_args)

        new_initial_allocation = period_allocation[0]

        # Update resource allocations and repeat SCO
        for inf_job, trg_job in current_best_config.items():
            res_alloc[inf_job][0] = new_initial_allocation[inf_job.name]
            if trg_job:
                res_alloc[inf_job][1] = new_initial_allocation[trg_job.name]
            else:
                res_alloc[inf_job][1] = 0   # No retraining done, allocate zero resources
        print("Iter {}, final_alloc: {}".format(i, res_alloc))

    return period_allocation

def fair_sco_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Picks hyperparameters according to the SCO procedure, but does not iteratively
    refine resource allocation like the thief scheduler
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Find best allocation
        best_trg_config, debug = single_camera_optimizer(training_configs, inference_job,
                                                         total_resources=total_resources,
                                                         retraining_period=retraining_period,
                                                         training_res_alloc=total_resources / 2,
                                                         inference_res_alloc=total_resources / 2)
        if best_trg_config:
            training_jobs.append(next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources)
    return period_allocation

def fair_naive_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Picks hyperparameters with highest accuracy that finish in the retraining period
    even though they may leave no time for exploitation.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Find naive allocation
        best_trg_config = single_camera_naive(training_configs, retraining_period, total_resources / 4)
        if best_trg_config:
            training_jobs.append(next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources)
    return period_allocation

def fair_random_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Picks hyperparameters randomly - for convinience picks the first in
    the list to avoid setting random seed - even though they may not complete on time.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    inference_jobs = [x[0] for x in job_pairs]
    training_jobs = [x[1][0] for x in job_pairs if len(x[1]) != 0]
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources)
    return period_allocation

def fair_dumb_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Dumb - picks the
    the list to avoid setting random seed - even though they may not complete on time.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Find naive allocation
        best_trg_config = single_camera_dumb(training_configs)
        if best_trg_config:
            training_jobs.append(next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources)
    return period_allocation

def inference_only_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Picks hyperparameters randomly - for convinience picks the first in
    the list to avoid setting random seed - even though they may not complete on time.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    inference_jobs = [x[0] for x in job_pairs]
    period_allocation = {0: {k.name: total_resources/len(inference_jobs) for k in inference_jobs}}
    return period_allocation

class simulator(object):
    '''
    A simulator class which runs for one retraining window of length retraining_period
    '''
    def __init__(self, training_jobs, inference_jobs, total_resources, quantum_size=1, retraining_period=100,
                 scheduling_algo=fair_scheduler, initial_allocation=None, period_allocation=None,
                 verbose=True, sim_name='unnamed'):
        self.training_jobs = {j.name: j for j in training_jobs}
        self.inference_jobs = {j.name: j for j in inference_jobs}
        self.total_resources = total_resources

        self.scheduling_algo = scheduling_algo

        self.current_t = 0
        self.quantum_size = quantum_size
        self.instantaneous_allocation = {}
        self.period_allocation = {} if not period_allocation else period_allocation
        self.initial_allocation = initial_allocation
        self.retraining_period = retraining_period

        self.logger = logging.getLogger(sim_name)
        self.logger.handlers = [logging.StreamHandler()]
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARN)

        if not self.period_allocation:
            self.compute_schedule()

        self.metrics = defaultdict(list)

    def compute_schedule(self):
        self.logger.info("No period allocation found, running scheduler.")
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
            if self.current_t % 50 == 0:
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

class MultiPeriodSimulator(object):
    '''
    Reuses the single period simulator and runs it multiple times, but updates job states across retraining windows.
    '''
    def __init__(self, cameras, total_resources, scheduler, retraining_period,
                 task_ids = ["1"], incomplete_jobs_checkpoint = False):
        self.incomplete_jobs_checkpoint = incomplete_jobs_checkpoint
        self.retraining_period = retraining_period
        self.cameras = cameras
        self.total_resources = total_resources
        self.scheduler = scheduler
        self.task_ids = task_ids


    def get_job_pairs(self, task_id):
        job_pairs = []
        train_to_camera_map = {}
        for camera in self.cameras:
            camera.generate_training_configurations(task_id)   # Generate new training configs
            train_configs = camera.get_training_configurations()
            job_pairs.append((camera.get_inference_job(), train_configs))
            for train_job in train_configs:
                train_to_camera_map[train_job.name] = camera
        return job_pairs, train_to_camera_map

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
        results = {}
        period_allocation_log = {}
        for task_id in self.task_ids:
            print("Running task {}".format(task_id))

            # Update all cameras to reset their inference accuracy to the one mentioned in profile. This is assuming models restart from scratch
            for camera in self.cameras:
                camera.reset_current_accuracy(task_id)

            # Get training profiles here
            job_pairs, train_to_camera_map = self.get_job_pairs(task_id)
            # Get new period allocation here given inference job and training profiles from Camera object
            period_allocation = self.scheduler(job_pairs, self.total_resources, self.retraining_period)
            period_allocation_log[task_id] = period_allocation

            training_jobs = []
            inference_jobs = []

            for jobname in period_allocation[0].keys():
                job = self.get_job_instance(job_pairs, jobname)
                if isinstance(job, InferenceJob):
                    inference_jobs.append(job)
                elif isinstance(job, TrainingJob):
                    training_jobs.append(job)
                else:
                    raise Exception
            sim = simulator(training_jobs, inference_jobs, self.total_resources,
                            retraining_period=self.retraining_period,
                            sim_name='period{}'.format(task_id),
                            period_allocation=period_allocation)
            results[task_id] = sim.step_till_completion()

            # Get metrics from all inference jobs and update the configurations in cameras and global result
            # Use training jobs to update accuracy or use inference jobs to update? Key question - do incomplete training jobs checkpoint at the end of retraining period?
            # for train_job in training_jobs:
            #     acc = train_job.get_current_accuracy()
            #     if train_job.is_done() or self.incomplete_jobs_checkpoint:
            #         camera = train_to_camera_map[train_job.name]
            #         camera.set_current_accuracy(acc)
            #     else:
            #         # Do not checkpoint if training job did not complete or
            #         pass
        results['meta'] = {'overall_inf_mean_auc': sum([results[r]['meta']['inf_mean_auc'] for r in results])/len(results)}
        return results, period_allocation_log

def generate_config(name, final_accuracy, job_time, start_accuracy, inference_job=None):
    if start_accuracy > final_accuracy:
        raise Exception("The model is already trained. start_accuracy {} > final_accuracy {}".format(start_accuracy,
                                                                                                     final_accuracy))
    init_time = 0
    func = lambda x: np.interp(x, [init_time, job_time], [start_accuracy, final_accuracy])
    return TrainingJob(name, func, init_time, job_time, resource_alloc=0, inference_job=inference_job)


def generate_config2(name, final_accuracy, job_time, start_accuracy=50, inference_job=None):
    # Starts from given accuracy
    if start_accuracy > final_accuracy:
        raise Exception("The model is already trained. start_accuracy {} > final_accuracy {}".format(start_accuracy,
                                                                                                     final_accuracy))
    # solve a=1 / ((1 / (t * 10 * 1 / (t+d))) + 1) * 100 * K, b=1 / ((1 / ((t+d) * 10 * 1 / (t+d))) + 1) * 100 * K for K, t
    a = start_accuracy
    b = final_accuracy
    d = job_time
    k = 11 * b / 1000
    t = a * d / (1000 * k - 11 * a)
    conv_time = t + d

    func, inv_func = get_optimus_fn(conv_time, k)
    init_time = inv_func(start_accuracy)
    return TrainingJob(name, func, init_time, conv_time, resource_alloc=0, inference_job=inference_job)

class Camera(object):
    def __init__(self, name, taskwise_training_profiles, taskwise_start_accuracy, inference_job=None,
                 inference_max_resources = 1, inference_camera_profile='c1', start_task_id = "1"):
        '''
        inference_job: InferenceJob object
        taskwise_start_accuracy: dict of {task_id: accs}. WARNING: How is the acc determined? Which Inference profile is it? That's preselected by the user right now..
        training_profiles: dict of {task_id: profge iles}. Each profile is a list of 3 tuples - [(final_acc, resource_time, start_acc), ..]
        '''
        self.name = name
        self.taskwise_training_profiles = taskwise_training_profiles
        self.taskwise_start_accuracy = taskwise_start_accuracy
        self.current_accuracy = taskwise_start_accuracy[start_task_id]
        self.configs = []
        if inference_job:
            self.inference_job = inference_job
        else:
            self.inference_job = InferenceJob("{}_inference".format(self.name),
                                              self.current_accuracy,
                                              get_infer_profile(max_inference_resources=inference_max_resources,
                                                                camera=inference_camera_profile)[0],
                                              #get_linear_fn(1/inference_max_resources)[0],
                                              resource_alloc=0)

    def get_training_configurations(self):
        return self.configs

    def generate_training_configurations(self, task_id):
        configs = []
        # Generates and returns the valid training configurations (final acc > current_acc)
        # TODO(romilb): Currently not bothering with profile curve, just start and end acc.
        for i, (final_acc, resource_time, _) in enumerate(self.taskwise_training_profiles[task_id]):
            if final_acc < self.current_accuracy:
                print("WARNING: Final accuracy {} less than current acc {} for camera {}, ignoring profile.".format(final_acc, self.current_accuracy, self.name))
            else:
                configs.append(generate_config("{}_train_{}".format(self.name, i),
                             final_acc,
                             resource_time,
                             self.current_accuracy,
                             self.inference_job))
        self.configs = configs
        return self.configs


    def get_inference_job(self):
        return self.inference_job

    def set_current_accuracy(self, acc):
        self.current_accuracy = acc
        self.inference_job.set_base_accuracy(self.current_accuracy)

    def reset_current_accuracy(self, task_id):
        # Sets accuracy to accuracy of given task
        self.current_accuracy = self.taskwise_start_accuracy[task_id]
        self.inference_job.set_base_accuracy(self.current_accuracy)