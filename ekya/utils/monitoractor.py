import time

import GPUtil

class Monitor(object):
    def __init__(self, node_id, logger_actor, gpu_idx = 0, log_period = 1, launch=True):
        self.node_id = node_id
        self.logger_actor = logger_actor
        self.log_period = log_period
        self.gpu_idx = gpu_idx
        self.fields = ["id", "load", "memoryTotal", "memoryUsed"]
        if launch:
            self.monitor_loop()

    def monitor_loop(self):
        while True:
            gpu = GPUtil.getGPUs()[0]
            data = [time.time()]
            for field in self.fields:
                data.append(getattr(gpu, field))
            self.logger_actor.append.remote(self.node_id + "_gpu_" + str(self.gpu_idx), data)
            self.logger_actor.flush.remote()
            time.sleep(self.log_period)