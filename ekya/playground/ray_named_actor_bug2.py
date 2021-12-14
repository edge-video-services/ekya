import ray
import time
@ray.remote
class A(object):
    def foo(self):
        time.sleep(5)
        return 1
@ray.remote
class B(object):
    def foo(self):
        return 2
ray.init()
a = A.options(name="my_actor").remote()
# Launch long task and immediately enqueue termination
task = a.foo.remote()
termination_task = a.__ray_terminate__.remote()
print("Long foo: {}".format(ray.get(task)))
# Ensure task termination before creating actor with same name
try:
    ray.get(termination_task)
except ray.exceptions.RayActorError:
    pass
local_handle = B.options(name="my_actor").remote() # Create new actor with same name
print("New foo from local handle: {}".format(ray.get(local_handle.foo.remote()))) # This works, my_actor is running.
ray_handle = ray.get_actor("my_actor")  # This fails with ValueError: The actor with name=my_actor is dead.
print("New foo from get_actor handle: {}".format(ray.get(ray_handle.foo.remote())))