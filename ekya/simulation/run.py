import copy
import os
from collections import defaultdict

import pandas as pd

from ekya.utils.helpers import read_json_file, write_json_file
from ekya.simulation.camera import Camera
from ekya.simulation.constants import INFER_MAX_RES_DICT, OPT
from ekya.simulation.jobs import InferenceJob
from ekya.simulation.schedulers import (fair_dumb_scheduler, fair_naive_scheduler,
                                        fair_random_scheduler, fair_sco_scheduler,
                                        inference_only_scheduler,
                                        thief_sco_scheduler, fair_fixedconfig_scheduler, cloud_scheduler)
# thief_scheduler,fair_scheduler,
from ekya.simulation.simulator import MultiPeriodSimulator

PROFILE_COLUMNS = ["timestamp", "train_time", "train_loss", "train_acc",
                   "train_num_batches", "val_time", "val_loss", "val_acc",
                   "val_num_batches", "test_time", "test_loss", "test_acc",
                   "test_num_batches"]


def read_profiles(profile_dir, city_name, use_oracle=False):
    # use_oracle: Use oracle data instead of predictions
    profile_path = os.path.join(profile_dir, f"{city_name}.json")
    data = read_json_file(profile_path)
    # offset = int(
    #     sorted(data["taskwise_train_profiles"], key=lambda x: int(x))[0])
    offset = 0
    final_data = {"taskwise_train_profiles": {},
                  "taskwise_infer_profiles": {},
                  "taskwise_predicted_profiles": {}}
    for key in data["taskwise_train_profiles"].keys():
        final_data["taskwise_train_profiles"][str(
            int(key)-offset)] = data["taskwise_train_profiles"][key]
    for key in data["taskwise_infer_profiles"].keys():
        final_data["taskwise_infer_profiles"][str(
            int(key)-offset)] = data["taskwise_infer_profiles"][key]
    for key in data["taskwise_predicted_profiles"].keys():
        if use_oracle:
            final_data["taskwise_predicted_profiles"][str(
                int(key)-offset)] = data["taskwise_train_profiles"][key]
        else:
            final_data["taskwise_predicted_profiles"][str(
                int(key)-offset)] = data["taskwise_predicted_profiles"][key]
    assert "taskwise_train_profiles" in final_data
    assert "taskwise_infer_profiles" in final_data
    return final_data


def get_citywise_data(results, city, retraining_period):
    tasks = list(results.keys())
    tasks.remove('meta')
    final_data = None
    for task in tasks:
        for job in results[task].keys():
            if isinstance(job, InferenceJob):
                if city in job.name:
                    d = results[task][job]
                    # interpolating the datapoints
                    i = 0
                    tmp_d = []
                    assert d[-1][0] == retraining_period
                    assert d[0][0] == 0
                    for k in range(int(d[-1][0]+1)):
                        if i < len(d) and k == d[i][0]:
                            item = copy.copy(d[i])
                            i += 1
                        else:
                            item = copy.copy(d[i-1])
                            item[0] = k
                        tmp_d.append(item)
                    d = tmp_d
                    time, acc, _ = list(zip(*d))
                    data = pd.DataFrame(
                        acc, columns=['task_{}'.format(task)], index=time)
                    data.index.name = 'time'
                    if final_data is None:
                        final_data = data
                    else:
                        final_data = final_data.join(data)
    return final_data


def create_cameras(city_names, profile_dir, real_inferece_profiles_path,
                   use_oracle=False):
    cameras = []
    real_inference_profiles = pd.read_csv(real_inferece_profiles_path)
    subsampling = real_inference_profiles['subsampling'].values
    cam = real_inference_profiles['c1'].values
    print("Creating cameras for cities {}".format(city_names))
    for idx, city in enumerate(city_names):
        data_pred = read_profiles(profile_dir, city, use_oracle)

        tasks = data_pred["taskwise_train_profiles"].keys()
        train_profiles = {}
        oracle_profiles = {}
        infer_start_profiles = {}
        for task_id in tasks:
            # Format = [Acc, res_time, hyp_id, model_name]
            train_profiles[task_id] = []
            for i, x in enumerate(
                    data_pred["taskwise_predicted_profiles"][task_id]):
                hyp_id, _ = x[0].split('_')  # change _ to epoch if needed
                train_profiles[task_id].append([x[2], x[3], x[0], x[4]])

            # Format = [Acc, res_time, hyp_id, model_name]
            oracle_profiles[task_id] = []
            for i, x in enumerate(
                    data_pred["taskwise_train_profiles"][task_id]):
                hyp_id, _ = x[0].split('_')  # change _ to epoch if needed
                oracle_profiles[task_id].append([x[2], x[3], x[0], x[4]])

            # TODO: Picking min profile, which else can we pick?
            # Format = [start Acc, model_name]
            config_accs = data_pred["taskwise_infer_profiles"][task_id]
            config_accs = list(config_accs.values())
            min_acc_profile = min(config_accs, key=lambda x: x[0])
            infer_start_profiles[task_id] = min_acc_profile

        cameras.append(
            Camera("{}_{}".format(city, idx), train_profiles, oracle_profiles,
                   infer_start_profiles, subsampling=subsampling,
                   inference_camera_profile=cam))
    return cameras


def run(args):
    print(args)
    generate_meta_profiles(args.root, args.camera_names, args.hyperparameters,
                           False, args.hyp_map_path)
    data = []
    allocation = defaultdict(dict)
    all_data = pd.DataFrame()
    all_results = defaultdict(dict)
    sched_tused_log = defaultdict(dict)
    periods_to_test = args.retraining_periods
    provisioned_res_to_test = args.provisioned_resources
    camera_cnt = len(args.camera_names)

    task_ids = [str(x) for x in range(1, args.num_tasks)]
    use_oracle_modes = [True]

    infer_max_res = min(INFER_MAX_RES_DICT.values())

    def generate_weighted_fair_schedulers():
        fair_scheduler_instance = fair_dumb_scheduler    # TODO: CHANGE THIS?
        weights = [0.25, 0.5, 0.7, 0.9, 1]
        schedulers = []
        for w in weights:
            # Currying to generate lambdas: https://stackoverflow.com/questions/53086592/lazy-evaluation-when-use-lambda-and-list-comprehension
            schedulers.append([(lambda w: lambda x, y, z: fair_scheduler_instance(x, y, z, inference_weight=w))(w), 'fair_dumb_{}'.format(w)])
        return schedulers


    def generate_cloud_schedulers(csv_delays):
        schedulers = []
        delays = [int(x) for x in csv_delays.split(',')]
        for d in delays:
            # Currying to generate lambdas: https://stackoverflow.com/questions/53086592/lazy-evaluation-when-use-lambda-and-list-comprehension
            schedulers.append([(lambda delay: lambda jobpairs, resources, ret_win: cloud_scheduler(jobpairs, resources, ret_win, cloud_delay=delay))(d), f'cloud_scheduler_{d}'])
        return schedulers

    schedulers_to_test = [
        # (lambda jobpairs, resources, ret_win: fair_fixedconfig_scheduler(
        #     jobpairs, resources, ret_win, config_id=args.fairfixed_config_id, config_epochs=args.fairfixed_config_epochs),
        #     'fairfixed_{}_{}'.format(args.fairfixed_config_id, args.fairfixed_config_epochs)),
        (fair_dumb_scheduler, 'fair_dumb'),
        (fair_naive_scheduler, 'fair_naive'),
        (fair_sco_scheduler, 'fair_sco'),
        (lambda jobpairs, resources, ret_win: thief_sco_scheduler(
            jobpairs, resources, ret_win, iterations=args.iterations,
            steal_increment=infer_max_res/args.iterations), 'thief'),
        (inference_only_scheduler, 'inference_only'),
        (fair_random_scheduler, 'fair_random')
    ]
    if args.cloud_delay:
        schedulers_to_test.extend(generate_cloud_schedulers(args.cloud_delay))
    schedulers_to_test.extend(generate_weighted_fair_schedulers())

    column_names = ['period', 'resources', 'use_oracle', *
                    [sched_name for _, sched_name in schedulers_to_test]]
    for provisioned_res in provisioned_res_to_test:
        for retraining_period in periods_to_test:
            for use_oracle in use_oracle_modes:
                this_config_result = [retraining_period,
                                      provisioned_res, use_oracle]
                for scheduler, sched_name in schedulers_to_test:
                    print(f"Sched: {sched_name}, Resources: {provisioned_res}."
                          f" Period: {retraining_period}. Use_Oracle: "
                          f"{use_oracle}.")
                    cameras = create_cameras(
                        args.camera_names, args.root,
                        args.real_inference_profiles, use_oracle=use_oracle)
                    mps = MultiPeriodSimulator(
                        cameras, provisioned_res, scheduler, retraining_period,
                        task_ids=task_ids, golden_model_delay=args.delay)
                    results, period_allocation_log, sched_tused = \
                        mps.step_till_completion()

                    # Get citywise data
                    for cam in cameras:
                        d = get_citywise_data(results, cam.name, retraining_period)
                        d['city'] = cam.name
                        d['res'] = provisioned_res
                        d['period'] = retraining_period
                        d['use_oracle'] = use_oracle
                        d['sched'] = sched_name
                        d['num_cams'] = len(args.camera_names)
                        all_data = pd.concat([all_data, d])
                    this_config_result.append(
                        results['meta']['overall_inf_mean_auc'])
                    key = "{}_{}_{}_{}".format(
                        retraining_period, provisioned_res, sched_name,
                        use_oracle)
                    allocation[key] = period_allocation_log
                    all_results[key] = {
                        r: results[r]['meta']['inf_mean_auc'] for r in results
                        if r != 'meta'}
                    sched_tused_log[key] = sched_tused

                data.append(this_config_result)
    # if len(args.camera_names) == 1:
    #     suffix = '{}_{}_{}delay{}'.format(
    #         args.dataset, args.camera_names[0], args.delay, "_opt" if OPT else "")
    # else:
    suffix = '{}_{}cam_{}delay{}'.format(
        args.dataset, camera_cnt, args.delay, "_opt" if OPT else "")

    # the schedulers' resource allocation over retraining period
    allocation_fname = os.path.join(
        args.output_path, f'allocation_{suffix}.json')

    # the schedulers' resource allocation over retraining period
    all_results_fname = os.path.join(
        args.output_path, f'results_{suffix}.json')

    # the schedulers' time used in schedule preparation
    tusage_fname = os.path.join(
        args.output_path, f'scheduler_time_usage_{suffix}.json')

    # the average accuracy comparison among different schedulers
    result_fname = os.path.join(
        args.output_path, f'scheduler_compare_result_{suffix}.csv')

    # the average accuracy comparison among different cameras/cities
    citywise_results_fname = os.path.join(
        args.output_path, f"citywise_results_{suffix}.csv")

    os.makedirs(args.output_path, exist_ok=True)
    write_json_file(allocation_fname, allocation)
    write_json_file(all_results_fname, all_results)
    write_json_file(tusage_fname, sched_tused_log)
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(result_fname, index=False)
    all_data.to_csv(citywise_results_fname)


def get_profile(json_path):
    data = read_json_file(json_path)
    profile_task_map = {}
    for taskid, profile_list in data.items():
        prof = pd.DataFrame(data[taskid], columns=PROFILE_COLUMNS)
        profile_task_map[taskid] = prof
    return profile_task_map


def get_subprofile_epochids(subprofile_json_path):
    data = read_json_file(subprofile_json_path)
    subprofiles_dict = list(data.values())[0]
    return [int(x) for x in subprofiles_dict.keys()]


def generate_meta_profiles(results_root, city_names, hyp_ids,
                           use_predictions=True, hyp_map_path="",
                           epochs_to_subprofile=[8, 17, 29]):
    """Generate subprofiles from profiling output.

    If use_predictions is false, writes oracle info as predictions."""
    profiles_dir = results_root
    os.makedirs(profiles_dir, exist_ok=True)
    print('The cameras are', city_names)
    print('Loading hyperparameter map from', hyp_map_path)
    hyp_map = read_json_file(hyp_map_path)

    for city in city_names:
        output_path = os.path.join(profiles_dir, f"{city}.json")
        city_path = os.path.join(results_root, city)
        hyperparam_ids = [str(hp_id) for hp_id in sorted(hyp_ids)]
        baseline_taskwise_acc = defaultdict(dict)
        train_taskwise_profiles = defaultdict(list)
        predicted_taskwise_profiles = defaultdict(list)
        for hp_id in hyperparam_ids:
            profile_path = os.path.join(city_path, f'{hp_id}_profile.json')
            # Not used and comment out for now
            # subprofile_path = os.path.join(
            #     city_path, '{}_subprofile_test.json'.format(hp_id))
            if use_predictions:
                try:
                    predictions_path = os.path.join(
                        city_path, '{}_predicted_acc_1.json'.format(hp_id))
                    predicted_profiles = read_json_file(predictions_path)
                except FileNotFoundError:
                    print(f'{predictions_path} does not exist. Skip.')
                    break
            taskwise_profiles = get_profile(profile_path)

            # For getting pre-retraining accuracy:
            retraining_result_path = os.path.join(
                city_path, '{}_retraining_result.json'.format(hp_id))
            retraining_result = read_json_file(retraining_result_path)
            preretrain_test_accs = retraining_result["preretrain_test_acc"]

            # epochs_to_subprofile = get_subprofile_epochids(subprofile_path)
            # [epoch ids where subprofiled]
            # Not used and comment out for now
            # untrained_accs = get_subprofile_default_accs(
            #     subprofile_path)  # {task_id: acc}
            task_ids = sorted(taskwise_profiles.keys(), key=lambda x: int(x))

            # print(epochs_to_subprofile)

            for task_id in [str(x) for x in task_ids[0:]]:
                profile = taskwise_profiles[task_id]
                if use_predictions:
                    prediction = predicted_profiles[str(int(task_id)-1)]

                # Get baseline no retraining inference accuracies
                task_start_acc = preretrain_test_accs[task_id]
                model_name = hyp_map[hp_id]['model_name']
                baseline_taskwise_acc[task_id][hp_id] = [
                    task_start_acc, model_name]
                for e in epochs_to_subprofile:
                    resource_time = sum(profile.loc[0:e]['train_time'])
                    start_acc = task_start_acc
                    actual_end_acc = profile.loc[e]['test_acc']
                    if use_predictions:
                        # ERROR - +1 wont exist always
                        predicted_end_acc = prediction[str(e)][task_id]
                    else:
                        predicted_end_acc = actual_end_acc
                    name = "{}_{}".format(hp_id, e)
                    train_taskwise_profiles[task_id].append(
                        [name, start_acc, actual_end_acc, resource_time,
                         model_name])
                    predicted_taskwise_profiles[task_id].append(
                        [name, start_acc, predicted_end_acc, resource_time,
                         model_name])
                    # TODO: ROMILB: Calculate end_acc by looking at
                    # task:task+1. Time is from task
        infer_train_profile = {
            'taskwise_train_profiles': train_taskwise_profiles,
            'taskwise_infer_profiles': baseline_taskwise_acc,
            'taskwise_predicted_profiles': predicted_taskwise_profiles}
        write_json_file(output_path, infer_train_profile)
