import ray
import time

@ray.remote
class A(object):
    def foo(self):
        time.sleep(20)
        return 1

ray.init()
act = A.remote()
act.foo()