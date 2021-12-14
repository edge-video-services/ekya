import base64
import os
import random
import time

from inclearn.models.icarl import ICarl
from inclearn.lib import data, utils, metrics
from ray.experimental import signal

from ekya.utils.helpers import timedrun
from ekya.utils.loggeractor import BaseLogger
import pickle

from ekya.utils.signals import IsAliveSignal


class ICaRLModel(object):
    def __init__(self, id=None, logger_actor=None, args=None, batch_size=None, subsample_dataset=None, debug=False):
        self.id = str(id) if id is not None else str(random.randint(0, 10000))
        self.model = ICarl(args)
        self.logger = logger_actor if logger_actor is not None else BaseLogger()
        self.handle = None
        self.debug = debug
        self.batch_size = batch_size if batch_size else 128
        self.subsample_dataset = subsample_dataset if subsample_dataset is not None else 1
        self.inc_dataset = data.IncrementalDataset(
            dataset_name="cifar100",
            random_order=False,
            shuffle=True,
            batch_size=self.batch_size,
            workers=0,
            subsample_dataset=subsample_dataset,
        )

    def set_handle(self, handle):
        self.handle = handle

    def infer(self, data_indexes):
        _, data_loader = self.inc_dataset.get_custom_index_loader(data_indexes, mode="test", shuffle=True)
        timetaken, [ypred, ytrue] = timedrun(self.model.eval_task, data_loader)
        accuracy = (ypred == ytrue).sum() / len(ytrue)
        acc_dict = metrics.accuracy(ypred, ytrue, task_size=10)
        self.logger.append.remote(self.id + "_infer_stats", [time.time(), timetaken, accuracy, len(ypred), base64.b64encode(pickle.dumps(acc_dict)).decode('utf-8')])
        if self.debug:
            print("Time taken to infer %d samples" % len(ypred))
            print(timetaken)
            print("accuracy to infer %d samples" % len(ypred))
            print(accuracy)
        self.logger.flush.remote()
        return ypred, ytrue, timetaken


    def infer_loop(self, low_class_idx, high_class_idx=None):
        signal_sent = False
        while True:
            self.infer(low_class_idx, high_class_idx)
            if not signal_sent:
                signal.send(IsAliveSignal(self.id))
                signal_sent = True


    def train(self, data_indexes, n_epochs=10, checkpoint_interval=0, checkpoint_path="/tmp/icarl_updated.pt"):
        if checkpoint_interval == 0:
            checkpoint_interval = n_epochs
        for _ in range(0,n_epochs, checkpoint_interval):
            result = self.train_atom(data_indexes, checkpoint_interval)
            if checkpoint_path is not None:
                self.checkpoint(checkpoint_path)
        return None

    def train_atom(self, data_indexes, n_epochs):
        start_time = time.time()
        _, train_loader = self.inc_dataset.get_custom_index_loader(data_indexes, mode="train", shuffle=True)

        self.model.eval()  # Set eval mode for pretrain
        timetaken_beforetask, _ = timedrun(self.model.before_task, train_loader, None)

        self.model.train()  # Set eval mode for train
        timetaken_traintask = 0
        for _ in range(0, n_epochs):
            print("Now processing epoch {}/{}".format(_, n_epochs))
            timetaken_trainepoch, [train_loss, val_loss] = timedrun(self.model.train_task, train_loader, None, n_epochs=1)
            timetaken_traintask += timetaken_trainepoch
            self.logger.append.remote(self.id + "_train_epochstats", [time.time(), timetaken_trainepoch, train_loss])
            self.logger.flush.remote()


        self.model.eval()
        timetaken_aftertask, _ = timedrun(self.model.after_task, train_loader)

        ypred, ytrue = self.model.eval_task(train_loader)
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=1)
        self.logger.append.remote(self.id + "_train_accuracy", [time.time(), str(acc_stats)])

        end_time = time.time()
        total_time = end_time - start_time


        self.logger.append.remote(self.id + "_train_times", [time.time(), total_time, timetaken_beforetask, timetaken_traintask, timetaken_aftertask])
        self.logger.flush.remote()
        #return acc_stats

    def checkpoint(self, path):
        self.model.checkpoint(path)

    def restore(self, path):
        if os.path.isfile(path):
            del self.model
            self.model = ICarl.from_checkpoint(path)
        else:
            print("WARNING: Checkpoint not found. Skipping restore..")

    def freeze_layers(self, layers_to_freeze = 7):
        self.model.freeze_layers(layers_to_freeze)

    def test_eval(self, data_indexes):
        _, test_loader = self.inc_dataset.get_custom_index_loader(data_indexes, mode="test", data_source="test", shuffle=True)
        ypred, ytrue = self.model.eval_task(test_loader)
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=1)
        self.logger.append.remote(self.id + "_test_accuracy", [time.time(), str(acc_stats)])
        self.logger.flush.remote()
        return acc_stats