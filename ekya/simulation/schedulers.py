import copy

import numpy as np
from numpy.testing import assert_almost_equal

from ekya.simulation.simulator import simulator, create_sim_env
from ekya.simulation.constants import TRAINING_COMPLETE_OVERHEAD


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


def thief_scheduler(training_jobs, inference_jobs, total_resources,
                    steal_increment=0.1, **kwargs):
    """Start off as a fair allocator, then "steal" resources from other jobs
    till you max out the accuracy. This may potentially require multiple nested
    simulators to evaluate each configuration.
    Return allocation for the entire retraining period as a dict with
    {timestamp: allocation}."""

    all_jobs = list(training_jobs.values()) + list(inference_jobs.values())
    # Initial fair allocation:
    fair_alloc = {j.name: total_resources/len(all_jobs) for j in all_jobs}

    best_period_allocation = {0: fair_alloc}

    # print("Init alloc: {}".format(period_allocation))

    # Iterate over all jobs to steal resources
    allocationid_to_score_map = {}
    allocationid_map = {}   # Maps all allocations to a hashable id
    alloc_id = -1

    # Add score for fair allocation:
    temp_sim = create_sim_env(training_jobs, inference_jobs,
                              fair_alloc, kwargs['retraining_period'],
                              total_resources, fair_scheduler)
    result = temp_sim.step_till_completion()
    allocationid_map[alloc_id] = temp_sim.period_allocation
    allocationid_to_score_map[alloc_id] = result['meta']['inf_mean_auc']
    alloc_id = 0

    for candidate_job in all_jobs:
        for victim_job in [x for x in all_jobs if x != candidate_job]:
            should_steal = True
            max_steal_resources = best_period_allocation[0][victim_job.name]
            steal_amounts = np.arange(0, max_steal_resources + steal_increment,
                                      steal_increment)
            steal_amounts = list(steal_amounts)
            for steal_amount in steal_amounts:
                if should_steal:
                    temp_period_allocation = copy.deepcopy(
                        best_period_allocation)
                    temp_curr_allocation = temp_period_allocation[0]
                    new_victim_handle = victim_job.name
                    new_candidate_handle = candidate_job.name
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

                    # New temp_allocation is ready, evaluate it's goodness with
                    # a simulator.
                    # TODO: is this bad? The simulator actually creates the
                    # best schedule.
                    temp_sim = create_sim_env(
                        training_jobs, inference_jobs, temp_curr_allocation,
                        kwargs['retraining_period'], total_resources,
                        fair_scheduler)
                    result = temp_sim.step_till_completion()

                    allocationid_map[alloc_id] = temp_sim.period_allocation
                    allocationid_to_score_map[alloc_id] = \
                        result['meta']['inf_mean_auc']
                    alloc_id += 1
            best_period_allocation = {0: allocationid_map[
                max(allocationid_to_score_map,
                    key=allocationid_to_score_map.get)][0]}

    # Pick the allocation with max score
    # print(allocationid_map)
    # print(allocationid_to_score_map)
    debug = [allocationid_map, allocationid_to_score_map]
    best_allocation = allocationid_map[max(
        allocationid_to_score_map, key=allocationid_to_score_map.get)]
    return best_allocation, debug


def fair_scheduler(training_jobs,
                   inference_jobs,
                   total_resources,
                   inference_weight=0.5,
                   **kwargs):
    """
    allocate each job equal amount of resources.
    Return allocation for the entire retraining period as a dict with
    {timestamp: allocation}."""
    all_jobs = list(training_jobs.values()) + list(inference_jobs.values())
    initial_allocation = kwargs.get('initial_allocation', {})
    if not initial_allocation:
        initial_allocation = {}
        inference_budget = total_resources*inference_weight
        training_budget = total_resources - inference_budget
        for j in list(training_jobs.values()):
            initial_allocation[j.name] = training_budget / len(list(training_jobs.values()))
        for j in list(inference_jobs.values()):
            initial_allocation[j.name] = inference_budget / len(list(inference_jobs.values()))
    allocation = {0: initial_allocation}

    # Perform reallocation when jobs complete

    period_allocation = fair_reallocator(
        training_jobs, inference_jobs, allocation)

    # Sanity check:
    for t, instant_alloc in period_allocation.items():
        try:
            assert_almost_equal(sum(instant_alloc.values()), total_resources,
                                err_msg="Got total resources: {}".format(
                sum(instant_alloc.values())))
        except Exception as e:
            print("WARNINGGG!")
            print(e)
    return period_allocation


def fair_reallocator(training_jobs, inference_jobs, initial_allocation):
    period_allocation = copy.copy(initial_allocation)
    running_training_jobs = [x for x in training_jobs.values()  # Shallow copy
                             if initial_allocation[0][x.name] != 0]
    while running_training_jobs:
        train_completion_times = {}  # Job: time
        for j in running_training_jobs:
            job_allocation = {t: alloc[j.name]
                              for t, alloc in period_allocation.items()}
            # add training job completion overhead
            train_completion_times[j] = j.completion_time(
                job_allocation) + TRAINING_COMPLETE_OVERHEAD
        # print(train_completion_times)
        completed_job, completion_time = min(
            train_completion_times.items(), key=lambda x: x[1])
        # print(f"Job {completed_job} completes at {completion_time}"

        last_allocation_time, last_instant_alloc = max(
            period_allocation.items(), key=lambda x: x[0])

        # Get resources of killed job and fairly redistribute
        running_training_jobs.remove(completed_job)
        current_jobs = running_training_jobs + \
            list(inference_jobs.values())  # .remove(completed_job)
        relinquished_resources = last_instant_alloc[completed_job.name]
        new_alloc_per_job = relinquished_resources / len(current_jobs)
        new_instant_alloc = {}
        for j in (list(set(training_jobs.values()) - set(running_training_jobs))):
            # Set allocation to zero for completed training jobs
            new_instant_alloc[j.name] = 0
        for j in current_jobs:
            new_instant_alloc[j.name] = last_instant_alloc[j.name] + \
                new_alloc_per_job  # Redistribute

        period_allocation[completion_time] = new_instant_alloc
    return period_allocation


def single_camera_optimizer(training_jobs, inference_job, total_resources,
                            retraining_period, training_res_alloc=0.5,
                            inference_res_alloc=0.5):
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
        # candidate_training_job = copy.deepcopy(training_job)
        # candidate_inference_job = copy.deepcopy(inference_job)
        candidate_training_job = copy.copy(training_job)
        candidate_inference_job = copy.copy(inference_job)
        candidate_inference_job.name = "inference_job"
        candidate_training_job.inference_job = candidate_inference_job

        initial_allocation = {0: {candidate_inference_job.name: inference_res_alloc,
                                  candidate_training_job.name: training_res_alloc}}
        period_allocation = fair_reallocator(
            {candidate_training_job.name: candidate_training_job},
            {candidate_inference_job.name: candidate_inference_job},
            initial_allocation)

        temp_sim = simulator([candidate_training_job],
                             [candidate_inference_job], total_resources,
                             fair_scheduler, quantum_size=1,
                             retraining_period=retraining_period,
                             period_allocation=period_allocation,
                             verbose=False)
        result = temp_sim.step_till_completion()

        allocationid_map[alloc_id] = period_allocation
        allocationid_to_score_map[alloc_id] = result['meta']['inf_mean_auc']
        alloc_id += 1

    best_allocation_jobnames = allocationid_map[max(
        allocationid_to_score_map, key=allocationid_to_score_map.get)][0].keys()
    training_job_name = next(
        x for x in best_allocation_jobnames if x != "inference_job")
    debug = [allocationid_to_score_map]
    return next(x for x in training_jobs if x.name == training_job_name), debug


def single_camera_naive(training_configs, retraining_period,
                        training_res_alloc, allow_no_train=True):
    '''
    A slightly dumb variant of Single Camera Optimizer. Picks a training
    configuration which fits in the retraining window
    and gives the max accuracy, even though does not leave enough time to
    exploit it.
    :param training_configs: The training configurations to pick from
    :return: training config with max accuracy that fits in the training window
    '''
    # Sort jobs by their completion accuracy and iterate over all
    # configurations to find with max accuracy that fits in retraining_period
    sorted_training_configs = sorted(
        training_configs, key=lambda x: x.get_completion_accuracy(),
        reverse=True)
    candidate_job = None
    for training_job in sorted_training_configs:
        trg_job_allocation = {0: training_res_alloc}
        completion_time = training_job.completion_time(trg_job_allocation)
        if completion_time < retraining_period:
            candidate_job = training_job
            break

    if candidate_job is None:
        if allow_no_train:
            print("No job found to fit in the retraining window. Not "
                  "retraining this period.")
            return None
        else:
            raise Exception("No job found to fit in the retraining window.")

    return candidate_job


def single_camera_dumb(training_configs):
    '''
    A dumb cloud-like scheduler which picks the config with max accuracy,
    irrespective of how much time it takes.
    :param training_configs: The training configurations to pick from
    :return: training config with max accuracy
    '''
    # Sort jobs by their completion accuracy and pick the first one.
    if len(training_configs) == 0:
        return None
    sorted_training_configs = sorted(
        training_configs, key=lambda x: x.get_completion_accuracy(),
        reverse=True)
    candidate_job = sorted_training_configs[0]
    return candidate_job


def thief_sco_scheduler(job_pairs, total_resources, retraining_period,
                        iterations=10, steal_increment=0.1):
    '''
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return:
    '''
    inference_trg_map = ReversibleDictionary(
        {inference_job: training_configs for inference_job, training_configs in job_pairs})
    current_best_config = {}

    # Set initial fair allocations
    per_job_alloc = total_resources/(len(inference_trg_map.keys())*2)
    # [inference, training] resources.
    res_alloc = {k: [per_job_alloc, per_job_alloc]
                 for k in inference_trg_map.keys()}

    for i in range(iterations):
        # First do SCO
        for inference_job, training_configs in inference_trg_map.items():
            best_trg_config_name, _ = single_camera_optimizer(
                training_configs, inference_job,
                total_resources=total_resources,
                retraining_period=retraining_period,
                inference_res_alloc=res_alloc[inference_job][0],
                training_res_alloc=res_alloc[inference_job][1]
            )
            if best_trg_config_name:
                best_trg_config = next(
                    x for x in training_configs if x.name == best_trg_config_name.name)
            else:
                best_trg_config = None
            current_best_config[inference_job] = best_trg_config
            print("Best config for {}: {}".format(
                inference_job.name, best_trg_config))
        training_jobs = list(
            [x for x in current_best_config.values() if x is not None])
        inference_jobs = list(current_best_config.keys())

        # Then run thief scheduler
        custom_args = {'retraining_period': retraining_period}
        period_allocation, _ = thief_scheduler(
            {t.name: t for t in training_jobs},
            {t.name: t for t in inference_jobs},
            total_resources, steal_increment=steal_increment, **custom_args)

        new_initial_allocation = period_allocation[0]

        # Update resource allocations and repeat SCO
        for inf_job, trg_job in current_best_config.items():
            res_alloc[inf_job][0] = new_initial_allocation[inf_job.name]
            if trg_job:
                res_alloc[inf_job][1] = new_initial_allocation[trg_job.name]
            else:
                # No retraining done, allocate zero resources
                res_alloc[inf_job][1] = 0.0
        print("Iter {}, alloc:".format(i))
        for j, r in res_alloc.items():
            print(f'\tJob {j.name}: infer:{r[0]}, train:{r[1]}')

    return period_allocation


def fair_sco_scheduler(job_pairs, total_resources, retraining_period, inference_weight=0.5):
    '''
    Picks hyperparameters according to the SCO procedure, but does not
    iteratively refine resource allocation like the thief scheduler
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Find best allocation
        best_trg_config, debug = single_camera_optimizer(
            training_configs, inference_job, total_resources=total_resources,
            retraining_period=retraining_period,
            training_res_alloc=total_resources / 2,
            inference_res_alloc=total_resources / 2)
        if best_trg_config:
            training_jobs.append(
                next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources, inference_weight=inference_weight)
    return period_allocation


def fair_naive_scheduler(job_pairs, total_resources, retraining_period, inference_weight=0.5):
    '''
    Picks hyperparameters with highest accuracy that finish in the retraining
    period even though they may leave no time for exploitation.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Find naive allocation
        best_trg_config = single_camera_naive(
            training_configs, retraining_period, total_resources / 4)
        if best_trg_config:
            training_jobs.append(
                next(a for a in training_configs
                     if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources, inference_weight=inference_weight)
    return period_allocation


def fair_random_scheduler(job_pairs, total_resources, retraining_period, inference_weight=0.5):
    '''
    Picks hyperparameters randomly - for convinience picks the first in the
    list to avoid setting random seed - even though they may not complete on
    time.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    inference_jobs = [x[0] for x in job_pairs]
    training_jobs = [x[1][0] for x in job_pairs if len(x[1]) != 0]
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources, inference_weight=inference_weight)
    return period_allocation

def fair_fixedconfig_scheduler(job_pairs, total_resources, retraining_period, config_id="0", config_epochs=17):
    '''
    Picks hyperparameters which have the specified config id and the config epochs.
    :param job_pairs: [(InferenceJob, [TrainingJob, TrainingJob ...]) ....]
    :param total_resources:
    :param retraining_period:
    :param config_id: Hyperparameter config id to use
    :param config_id: Hyperparameter epochs to use
    :return: period allocation
    '''
    inference_jobs = [x[0] for x in job_pairs]
    all_configs = [x[1] for x in job_pairs if len(x[1]) != 0]
    training_jobs = []
    #training_jobs = [[cfg for cfg in cam_cfgs if cfg.name == "{}_{}".format(config_id, config_epochs)] for cam_cfgs in all_configs]  # cfg[0] is the name in the format hpid_epochs. Cfg is [name, start_acc, predicted_end_acc, resource_time, model_name]
    for cam_cfgs in all_configs:
        config_found = False
        for cfg in cam_cfgs:
            # cfg type is TrainingJob. Name format is '{city}_{id}_train_{hpid}_{epochs}'
            epochs = cfg.name.split('_')[-1]
            hp_id = cfg.name.split('_')[-2]
            if epochs == str(config_epochs) and hp_id == str(config_id):
                training_jobs.append(cfg)
                config_found = True
        if not config_found:
            print("[FairFixedConfig Sched] WARNING: Config {}, {} not found for {}".format(config_id, config_epochs, cam_cfgs[0].name))
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources)
    return period_allocation

def fair_dumb_scheduler(job_pairs, total_resources, retraining_period, inference_weight=0.5):
    '''
    Dumb - picks the
    the list to avoid setting random seed - even though they may not complete
    on time.
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
            training_jobs.append(
                next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)
    period_allocation = fair_scheduler({j.name: j for j in training_jobs},
                                       {j.name: j for j in inference_jobs},
                                       total_resources, inference_weight=inference_weight)
    return period_allocation

def cloud_scheduler(job_pairs, total_resources, retraining_period, cloud_delay=0):
    '''
    Scheduler to simulate the cloud
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    training_jobs = []
    inference_jobs = []
    for inference_job, training_configs in job_pairs:  # Use single camera dumb to find the highest performing configs
        best_trg_config = single_camera_dumb(training_configs)
        if best_trg_config:
            training_jobs.append(
                next(a for a in training_configs if a.name == best_trg_config.name))
        inference_jobs.append(inference_job)

    # Initial allocation is all resources fairly split to inference
    initial_allocation = {}
    for j in training_jobs:
        initial_allocation[j.name] = 0.000001   # Not setting to zero to avoid ignoring this job in the simulator
    for j in inference_jobs:
        initial_allocation[j.name] = total_resources / len(inference_jobs)

    # After cloud delay, assign nearly infinite resources to training so that jobs train instantly
    delayed_allocation = {}
    for j in training_jobs:
        delayed_allocation[j.name] = 999999   # Train instantly
    for j in inference_jobs:
        delayed_allocation[j.name] = total_resources / len(inference_jobs)

    period_allocation = {0: initial_allocation,
                         cloud_delay: delayed_allocation}

    # No need to run reallocator since the training jobs finish on the first step and inference jobs continue with old allocations
    return period_allocation

def inference_only_scheduler(job_pairs, total_resources, retraining_period):
    '''
    Picks hyperparameters randomly - for convinience picks the first in the
    list to avoid setting random seed - even though they may not complete on
    time.
    :param job_pairs: [(Inference, [training_configs]) ....]
    :param total_resources:
    :param retraining_period:
    :return: period allocation
    '''
    inference_jobs = [x[0] for x in job_pairs]
    period_allocation = {0: {k.name: total_resources /
                             len(inference_jobs) for k in inference_jobs}}
    return period_allocation
