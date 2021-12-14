import math

from ekya.classes.model import DEFAULT_HYPERPARAMETERS


def prepare_hyperparameters(hyps: dict) -> [dict]:
    '''
    Cleans up a hyperparameter dict to be used in Ekya. Creates keys if necessary.
    :param hyps: dict of raw hyperparameters
    :return: dict of cleaned up hyperparameters
    '''
    if "train_batch_size" not in hyps:
        hyps["train_batch_size"] = hyps["batch_size"] // 8
    if "test_batch_size" not in hyps:
        hyps["test_batch_size"] = DEFAULT_HYPERPARAMETERS["test_batch_size"]
    if "num_classes" not in hyps:
        hyps["num_classes"] = DEFAULT_HYPERPARAMETERS["num_classes"]
    # if True:
    #     if hyps["model_name"] == "resnet101":
    #         print("[WARN] Modifying hyperparameters to always use resnet18!")
    #         hyps["model_name"] = "resnet18"
    return hyps

def convert_to_ray_demands(inference_memory_demand: dict,
                               inference_resource_weights: dict,
                               training_memory_demand: dict,
                               training_resource_weights: dict):
    # TODO: This is trash design, change to accept just two dictionaries and invoke this fn twice.
    # Ray randomly allocates GPUs so we have to work around it by setting resource requirements on GPUs. This method returns resources that must be requested by tasks from Ray.
    ray_inference_resource_demands = {}
    ray_training_resource_demands = {}

    # Scale resource weights from 0-100 to 0-1
    i_wts = {k: v/100 for k,v in inference_resource_weights.items()}
    t_wts = {k: v/100 for k,v in training_resource_weights.items()}

    for cameraid in i_wts:
        if i_wts[cameraid] != 0:
            ray_wt = max(i_wts[cameraid], inference_memory_demand[cameraid])
        else:
            ray_wt = 0
        ray_inference_resource_demands[cameraid] = ray_wt
    for cameraid in t_wts:
        if t_wts[cameraid] != 0:
            ray_wt = max(t_wts[cameraid], training_memory_demand[cameraid])
        else:
            ray_wt = 0
        ray_training_resource_demands[cameraid] = ray_wt

    return ray_inference_resource_demands, ray_training_resource_demands

def quantize_demands(ray_resource_demands):
    # Quantizes allocations to 1/n, where n is integer
    # Algo - invert the fraction, then take ciel and invert again.
    # TODO: Does not account what to do with surplus
    quantized_demands = {}
    for camera_id, demand in ray_resource_demands.items():
        if demand == 0:
            quantized_demands[camera_id] = 0
        else:
            quantized_demands[camera_id] = 1/(math.ceil(1/demand))
    return quantized_demands