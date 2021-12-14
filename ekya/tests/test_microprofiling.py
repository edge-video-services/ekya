import time
from collections import defaultdict

import numpy as np
import json
import os
import pandas as pd

import ray

from ekya.classes.model import DEFAULT_HYPERPARAMETERS, RayMLModel
from ekya.microprofilers.modelling_funcs import get_scaled_optimus_fn, DEFAULT_SCALED_OPTIMUS_ARGS, get_linear_fn
from ekya.microprofilers.runtime_data import MEASURED_TIME_PER_EPOCH, MEASURED_INITTIME
from ekya.microprofilers.simple_microprofiler import SimpleMicroprofiler, subsample_dataloader
from ekya.classes.camera import Camera

LOCAL_RUN = False
if __name__ == '__main__':
    if LOCAL_RUN:
        print("Running locally with reduced params!")
    DF_HEADERS = ["city", "task_id", "hp_id", "epochs", "accuracy_pred", "runtime_pred", "accuracy_actual",
                  "runtime_actual", "default_inference_acc_pred", "default_inference_acc_actual",
                  "microprofile_time_taken",
                  "runtime_inittime_pred", "runtime_timeperepoch_pred", "runtime_inittime_actual", "runtime_timeperepoch_actual"]
    CITIES = ["zurich", "hamburg", "erfurt", "cologne", "stuttgart", "hanover", "strasbourg", "bremen", "aachen",
              "monchengladbach", "frankfurt", "lindau", "tubingen", "bochum", "ulm", "weimar", "darmstadt", "krefeld",
              "jena", "dusseldorf", "munster"] if not LOCAL_RUN else ["zurich", "bremen"]
    MICROPROFILE_NUM_EPOCHS = 5
    subsample = 0.1
    TASK_IDS = range(1, 4) if not LOCAL_RUN else [2]
    PROFILE_EPOCHS = np.array([5, 15, 30]) if not LOCAL_RUN else np.array([5, 10])
    dataset_root = '/ekya/datasets/cityscapes/'
    model_path = '/ekya/models/'
    pretrained_path = '/ekya/models/pretrained_cityscapes_fftmunster_{}_{}x2.pt'
    hyps_path = '/ekya/ekya/ekya/experiment_drivers/utilitysim_schedules/hyp_map_18only.json'
    inference_profile_path = '/ekya/ekya/ekya/experiment_drivers/real_inference_profiles.csv'
    microprofile_predmodel_acc_args = DEFAULT_SCALED_OPTIMUS_ARGS
    with open(hyps_path) as f:
        HYPERPARAMS = json.load(f)
    hyp_list = list(HYPERPARAMS.values())
    hyp_list = hyp_list if not LOCAL_RUN else hyp_list[0:2]
    for h in hyp_list:
        h['num_classes'] = 6
    expt_results = []
    for CITY in CITIES:
        ray.init(num_cpus=10)
        camera = Camera("test",
                        train_sample_names=[CITY],
                        sample_list_path=os.path.join(dataset_root, 'sample_lists', 'citywise'),
                        num_tasks=10,
                        train_split=0.9,
                        pretrained_sample_names=["frankfurt", "munster"],
                        dataset_name='cityscapes',
                        dataset_root=dataset_root,
                        max_inference_resources=0.25)

        for TASK_ID in TASK_IDS:
            dataloaders = [camera._get_dataloader(task_id=TASK_ID, train_batch_size=16, test_batch_size=16,
                                                  subsample_rate=hp["subsample"]) for hp in hyp_list]
            mp = SimpleMicroprofiler()
            candidate_hyperparams = hyp_list
            microprofile_start_time = time.time()
            best_result, microprofile_results = mp.run_microprofiling(candidate_hyperparams,
                                                                      dataloaders,
                                                                      resources=0.8,
                                                                      epochs=MICROPROFILE_NUM_EPOCHS,
                                                                      pretrained_model_format=pretrained_path,
                                                                      subsample_rate=subsample
                                                                      )

            # Get inference results
            default_inference_accs = {
                camera.id: [hp_result['preretrain_test_acc'] for hp_result in microprofile_results]}

            # Run modelling
            camera_profiles = defaultdict(dict)
            unsuccesful_models = 0
            for hp_result, default_acc in zip(microprofile_results, default_inference_accs[camera.id]):
                test_acc, hyperparameters, init_time, time_per_epoch = hp_result['test_acc'], hp_result[
                    'hyperparameters'], hp_result['init_time'], hp_result['time_per_epoch']
                try:
                    microprofile_accuracy_model = get_scaled_optimus_fn(
                        microprofile_x=np.array([MICROPROFILE_NUM_EPOCHS]),
                        microprofile_y=np.array([test_acc]),
                        start_acc=default_acc,
                        **microprofile_predmodel_acc_args)
                except RuntimeError:
                    unsuccesful_models += 1
                    # Simply return the start accuracy
                    microprofile_accuracy_model = lambda x: default_acc * np.ones_like(x)
                time_per_epoch = MEASURED_TIME_PER_EPOCH[hyperparameters['id']]
                init_time = MEASURED_INITTIME[hyperparameters['id']]
                microprofile_runtime_model = get_linear_fn(a=time_per_epoch,
                                                           b=init_time)
                acc_predictions = microprofile_accuracy_model(PROFILE_EPOCHS)
                runtime_predictions = microprofile_runtime_model(PROFILE_EPOCHS)
                for acc_prediction, runtime_prediction, epochs in zip(acc_predictions, runtime_predictions,
                                                                      PROFILE_EPOCHS):
                    camera_profiles[hyperparameters['id']][epochs] = {'accuracy': acc_prediction,
                                                                      'runtime': runtime_prediction,
                                                                      'pre_retrain_acc': default_acc,
                                                                      'runtime_init_time': hp_result['init_time'],
                                                                      'runtime_unscaled_time_per_epoch': hp_result[
                                                                          'time_per_epoch']}

            if unsuccesful_models:
                print(
                    "[THIEF SCHEDULER][WARN] Failed to generate models for {} cameras. Using default inference accuracy.".format(
                        unsuccesful_models))

            microprofile_time_taken = time.time() - microprofile_start_time
            print("Microprofiling done in {}, now evaluating actuals".format(microprofile_time_taken))
            print(camera_profiles)

            # Measure accuracy without microprofiling
            camera.set_current_task(1)
            camera.inference_gpu_weight = 100
            camera_actuals = defaultdict(dict)
            for hp in hyp_list:
                for ep_to_test in PROFILE_EPOCHS:
                    hp = hp.copy()
                    hp["epochs"] = ep_to_test
                    dataloaders = camera._get_dataloader(task_id=TASK_ID, train_batch_size=16, test_batch_size=16,
                                                         subsample_rate=hp["subsample"])
                    restore_path = pretrained_path.format(hp["model_name"], hp["num_hidden"])
                    start_time = time.time()
                    retraining_task = camera.run_retraining(hyperparameters=hp,
                                                            training_gpu_weight=100,
                                                            dataloaders_dict=dataloaders,
                                                            restore_path=restore_path,
                                                            profiling_mode=False)
                    best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_return = ray.get(
                        retraining_task)
                    runtime_actual = time.time() - start_time
                    inference_task = camera.training_model.test_acc.remote(test_loader=dataloaders["test"],
                                                                           resource_scaled=False)
                    accuracy_actual = ray.get(inference_task)
                    camera_actuals[hp['id']][ep_to_test] = {'accuracy': accuracy_actual,
                                                            'runtime': runtime_actual,
                                                            'pre_retrain_acc': profile_preretrain_test_acc,
                                                            'runtime_init_time': misc_return['init_time'],
                                                            'runtime_unscaled_time_per_epoch': misc_return[
                                                                'per_epoch_avg_time']}

            # Accumulate the results
            for hp in hyp_list:
                for ep_to_test in PROFILE_EPOCHS:
                    hp_id = hp['id']
                    accuracy_pred = camera_profiles[hp['id']][ep_to_test]['accuracy']
                    runtime_pred = camera_profiles[hp['id']][ep_to_test]['runtime']
                    default_acc_pred = camera_profiles[hp['id']][ep_to_test]['pre_retrain_acc']
                    runtime_inittime_pred = camera_profiles[hp['id']][ep_to_test]['runtime_init_time']
                    runtime_timeperepoch_pred = camera_profiles[hp['id']][ep_to_test]['runtime_unscaled_time_per_epoch']
                    accuracy_actual = camera_actuals[hp['id']][ep_to_test]['accuracy']
                    runtime_actual = camera_actuals[hp['id']][ep_to_test]['runtime']
                    default_acc_actual = camera_actuals[hp['id']][ep_to_test]['pre_retrain_acc']
                    runtime_inittime_actual = camera_actuals[hp['id']][ep_to_test]['runtime_init_time']
                    runtime_timeperepoch_actual = camera_actuals[hp['id']][ep_to_test]['runtime_unscaled_time_per_epoch']

                    data = [CITY, TASK_ID, hp_id, ep_to_test, accuracy_pred, runtime_pred, accuracy_actual,
                            runtime_actual, default_acc_pred, default_acc_actual, microprofile_time_taken,
                            runtime_inittime_pred, runtime_timeperepoch_pred, runtime_inittime_actual, runtime_timeperepoch_actual]
                    expt_results.append(data)

            df = pd.DataFrame(expt_results)
            df.index.name = 'index'
            df.to_csv('/tmp/ekya_microprofiling_bench.csv', index=False, header=DF_HEADERS)
        ray.shutdown()
