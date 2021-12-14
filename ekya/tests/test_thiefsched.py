import json
import os
import time

import ray

from ekya.microprofilers.modelling_funcs import DEFAULT_SCALED_OPTIMUS_ARGS
from ekya.models.hyperparameters import DEFAULT_HYPERPARAMETERS
from ekya.schedulers.thief_scheduler import ThiefScheduler
from ekya.classes.camera import Camera



if __name__ == '__main__':
    ray.init(num_cpus=10)
    NUM_EPOCHS = 1
    NUM_CAMS = 2
    dataset_root = '/ekya/datasets/cityscapes/'
    model_path = '/ekya/models/'
    pretrained_path = '/ekya/models/pretrained_cityscapes_fftmunster_{}_{}x2.pt'
    hyps_path = '/ekya/ekya/ekya/experiment_drivers/utilitysim_schedules/hyp_map_18only.json'
    inference_profile_path = '/ekya/ekya/ekya/experiment_drivers/real_inference_profiles.csv'
    with open(hyps_path) as f:
        HYPERPARAMS = json.load(f)
    hyp_list = list(HYPERPARAMS.values())
    for h in hyp_list:
        h['num_classes'] = 6
    cameras = []
    for cameras_count in range(0, NUM_CAMS):
        camera = Camera(f"test{cameras_count}",
                        train_sample_names=["zurich"],
                        sample_list_path=os.path.join(dataset_root, 'sample_lists', 'citywise'),
                        num_tasks=10,
                        train_split=0.9,
                        pretrained_sample_names=["frankfurt", "munster"],
                        dataset_name='cityscapes',
                        dataset_root=dataset_root,
                        max_inference_resources=0.25)
        camera.update_inference_model(hyperparameters=DEFAULT_HYPERPARAMETERS,
                                      inference_gpu_weight=100,
                                      ray_resource_demand=0.01,
                                      restore_path=pretrained_path.format(DEFAULT_HYPERPARAMETERS['model_name'], DEFAULT_HYPERPARAMETERS['num_hidden']))
        cameras.append(camera)

    microprofile_predmodel_acc_args = DEFAULT_SCALED_OPTIMUS_ARGS
    sched = ThiefScheduler(hyps_path=hyps_path,
                           inference_profile_path=inference_profile_path,
                           pretrained_model_dir=model_path,
                           microprofile_resources_per_trial=0.7,
                           microprofile_epochs=2,
                           microprofile_device='cuda',
                           predmodel_acc_args=microprofile_predmodel_acc_args)
    sched_state = {
        "task_id": 1,
        "retraining_period": 200,
        "test_subsample_rate": 0.1
    }
    start_time = time.time()
    sched = sched.get_schedule(cameras, resources=1, state=sched_state)
    end_time = time.time()
    print(sched)
    print("Time taken - {:.2f}".format(end_time-start_time))
