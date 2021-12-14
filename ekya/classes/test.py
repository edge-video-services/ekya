from ekya.classes.model import MLModel

import ray
ray.init()
rml = ray.remote(MLModel)

model_args = {"num_classes": 1}
hyps = {"num_hidden": 10,
        "last_layer_only": False,
        "model_name": "resnet18"}
results = {}
handles = {}
for i in range(1,5):
    handles[i] = rml.remote(model_args, hyperparameters=hyps, gpu_allocation=i/10)
    results[i] = ray.get(handles[i].get_pid.remote())

print(results)