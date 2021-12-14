import random
import time
import numpy as np

import ray
from inclearn.lib import data

class InferenceManager(object):
    def __init__(self, id, model_actor, logger_actor, batch_size=32, contention_slowdown = 0.5):
        self.id = str(id) if id is not None else str(random.randint(0, 10000))
        self.model_actor = model_actor
        self.logger_actor = logger_actor
        self.timebook = {"start": {}, "end": {}}
        self.running_request = None
        self.my_handle = None   # Actor handle to invoke tasks on self
        self.batch_size = batch_size

        # Is under contention? Used when reporting accuracy
        self.contention = False
        self.contention_slowdown = contention_slowdown
        self.run = True

        self.setup_data()

    def setup_data(self):
        # Initialize a dataset object to get inference indices
        self.dataset = data.IncrementalDataset(
                        dataset_name="cifar100",
                        batch_size=1,
                        workers=0,
                        increment=1,
                        is_sampleincremental=True
                    )
        self.num_samples = self.dataset.data_test.shape[0]
        self.sample_idxs = np.arange(0, self.num_samples)
        np.random.shuffle(self.sample_idxs)
        self.current_batch_idx = 0

    def get_next_idxes(self):
        if (self.current_batch_idx+1)*self.batch_size > self.num_samples:
            self.current_batch_idx = 0
        idxs = self.sample_idxs[self.current_batch_idx*self.batch_size:(self.current_batch_idx+1)*self.batch_size]
        self.current_batch_idx += 1
        if self.contention:
            # Send a subset of the of elements for inference beacuse subsampling
            idxs = idxs[0:int(len(idxs)*self.contention_slowdown)]
        return idxs

    def infer(self, data_indexes):
        # Wait for any pending requests to complete before issuing a new one
        if self.running_request is not None:
            print("Waiting for pending inference to complete")
            ypred, ytrue, timetaken = ray.get([self.running_request])[0]
            #print(result)
            time.sleep(1)
            self.timebook["end"][self.running_request] = time.time()
            accuracy = (ypred == ytrue).sum() / len(ytrue)
            if self.contention:
                accuracy = accuracy*self.contention_slowdown
            print("Accuracy of last batch: {}".format(accuracy))
            self.logger_actor.append.remote("infer_mgr_" + self.id + "_infer_stats", [time.time(), timetaken,
                                                                                self.timebook["start"][self.running_request],
                                                                                self.timebook["end"][self.running_request],
                                                                                      accuracy,
                                                                                len(ypred)])

        # Run next inference request
        print("Placing inference request")
        self.running_request = self.model_actor.infer.remote(data_indexes)
        self.timebook["start"][self.running_request] = time.time()

    def run_inference_loop(self):
        if self.run:
            data_indexes = self.get_next_idxes()
            print("Using data indexes: {}".format(data_indexes))
            self.my_handle.infer.remote(data_indexes)
            self.my_handle.run_inference_loop.remote()
        return None

    def stop(self):
        print("Stopping inference manager.")
        self.run = False

    def resume(self):
        print("Resuming inference manager.")
        self.run = True

    def set_contention(self, contention):
        print("Setting contention to {}".format(contention))
        self.contention = contention

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def register_handle(self, handle):
        self.my_handle = handle

    def restore(self, checkpoint_path):
        ray.get(self.model_actor.restore.remote(checkpoint_path))
