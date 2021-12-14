import time

import ray

@ray.remote
class runner(object):
    def step(self):
        print("In runner, running.")
        time.sleep(1)   # pretend work
        return 0

@ray.remote
class worker(object):
    def __init__(self, runner_actor):
        self.runner_actor = runner_actor
        self.value = 0
        self.run = True
        self.counter = 0

        self.running_task = None

    def register_handle(self, handle):
        self.handle = handle

    def work(self):
        if self.running_task is not None:
            ray.get([self.running_task])    # wait for tasks to complete
        self.running_task = self.runner_actor.step.remote()
        self.counter += 1
        print("Value: {}".format(self.value))

    def stop(self):
        self.run = False
        print("Counter: {}".format(self.counter))

    def resume(self):
        self.run = True

    def update_value(self, value):
        print("Updating value to {}".format(value))
        self.value = value

    def do_work(self):
        if self.run:
            self.handle.work.remote()
            self.handle.do_work.remote()
        return None

if __name__ == '__main__':
    ray.init(num_cpus=10)
    r = runner.remote()
    w = worker.remote(r)
    ray.get(w.register_handle.remote(w))
    w.do_work.remote()
    time.sleep(1)
    w.update_value.remote(1)
    w.update_value.remote(2)
    w.update_value.remote(3)
    w.update_value.remote(4)
    w.update_value.remote(5)
    time.sleep(3)
    ray.get(w.stop.remote())