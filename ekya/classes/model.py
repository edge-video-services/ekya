import time

from ekya.models.resnet import Resnet
from ekya.utils.mps import set_mps_envvars
from ekya.CONFIG import RANDOM_SEED
from ekya.utils.helpers import seed_all
import numpy as np
import torch
import ray
import os

DEFAULT_HYPERPARAMETERS = {
    'num_classes': 6,
    'epochs': 3,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'num_hidden': 512,
    'last_layer_only': False,
    'model_name': "resnet18",
    'train_batch_size': 16,
    'test_batch_size': 16
}

# Set random seed for reproducibility
seed_all(RANDOM_SEED)

class MLModel(object):
    def __init__(self,
                 hyperparameters: dict,
                 gpu_allocation_percentage: float,
                 inference_scaling_function: callable = lambda x: 1,
                 restore_path: str = "",
                 device: str = 'auto',
                 name='unnamed'):
        self.hyperparameters = hyperparameters
        self.gpu_allocation_percentage = gpu_allocation_percentage
        self.restore_path = restore_path
        set_mps_envvars(gpu_allocation_percentage)
        print("Initializing {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))
        NUM_CLASSES = self.hyperparameters["num_classes"]
        self.inference_scaling_function = inference_scaling_function    # The input range of the inference scaling function should be 0-1. Need to translate percentage by /100.
        self.name = name
        self.model = Resnet(NUM_CLASSES, hyperparameters=self.hyperparameters, restore_path=self.restore_path, device=device)

    def retrain_model(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader,
                      hyperparameters: dict,
                      validation_freq: int = -1,
                      profiling_mode = False):
        """
        Retrains a model given new dataloader.
        :param train_loader: Dataloader for the train set
        :param val_loader: Dataloader for the validation set
        :return:
        """
        total_start_time = time.time()
        NUM_EPOCHS = hyperparameters["epochs"]
        LR = hyperparameters["learning_rate"]
        MOMENTUM = hyperparameters["momentum"]

        if validation_freq == -1:
            validation_freq = NUM_EPOCHS

        dataloaders_dict = {'train': train_loader,
                            'val': val_loader,
                            'test': test_loader}

        profile_preretrain_test_acc = None
        profile_test_acc = None

        print("Retraining hyps: {}\nHashes:\nTrain: {}\nVal: {}\nTest: {}".format(hyperparameters,
                                                                                  train_loader.dataset.get_md5(),
                                                                                  test_loader.dataset.get_md5(),
                                                                                  val_loader.dataset.get_md5()))
        #if profiling_mode:
        start_time = time.time()
        profile_preretrain_test_acc = self.model.infer(dataloaders_dict['test'])
        infer_time_pre = time.time() - start_time
        print("Profile mode: pre-retrain testing took {} seconds and got acc {:.2f}".format(infer_time_pre, profile_preretrain_test_acc))
        #print("Dataloader: {}".format(dataloaders_dict['test'].dataset.samples))
        if infer_time_pre > 5:
            print("[WARNING] Inference is taking too long - make preretrain and post retrain testing only in profile mode.")

        retrain_start_time = time.time()
        _, _, best_val_acc, profile, subprofile_test_results, misc_return = self.model.train_model(dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR,
                                                                                      momentum=MOMENTUM, validation_freq=validation_freq)
        retrain_end_time = time.time()
        retrain_time = retrain_end_time - retrain_start_time

        #if profiling_mode:
        start_time = time.time()
        profile_test_acc = self.model.infer(dataloaders_dict['test'])
        infer_time_post = time.time() - start_time
        print("Profile mode: post-retrain testing took {} seconds and got acc {:.2f}".format(infer_time_post, profile_test_acc))
        #print("Dataloader: {}".format(dataloaders_dict['test'].dataset.samples))
        if infer_time_post > 5:
            print("[WARNING] Inference is taking too long - make preretrain and post retrain testing only in profile mode.")
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        misc_return['total_time'] = total_time
        misc_return['init_time'] = total_time - NUM_EPOCHS*misc_return['per_epoch_avg_time']

        return best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_return

    def test_acc(self, test_loader: torch.utils.data.DataLoader, resource_scaled=True):
        test_acc = self.model.infer(test_loader)
        # Implement scaling with GPU allocation here.
        if resource_scaled:
            scaling_factor = self.inference_scaling_function(self.gpu_allocation_percentage/100)
            print("[INFERENCE DEBUG] GPU weight = {}. Scaling factor: {}. Orig Accuracy = {}.".format(self.gpu_allocation_percentage/100,
                                                                                                      scaling_factor,
                                                                                                      test_acc))
            scaled_test_acc = scaling_factor * test_acc
        else:
            scaled_test_acc = test_acc
        return scaled_test_acc

    def save_model(self, path):
        '''Checkpoint to disk.'''
        self.model.save(path)

    def get_pid(self):
        print("I am PID: {}".format(os.getpid()))
        print("Envvar: {}".format(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "Not Defined")))
        return os.getpid(), os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "Not Defined")

    def ready(self):
        '''
        Dummy function to signal actor is ready.
        :return:
        '''
        return True

    def get_gpu_allocation(self):
        return self.gpu_allocation_percentage

RayMLModel = ray.remote(num_gpus=0.01)(MLModel)