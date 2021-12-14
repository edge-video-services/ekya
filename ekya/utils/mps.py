import os


def set_mps_envvars(gpu_allocation: int):
    '''
    Sets the environment variables for MPS.
    :param gpu_allocation: thread allocation as an int 0-100.
    '''
    assert 0 < gpu_allocation, "Invalid GPU allocation: {}".format(gpu_allocation)
    if gpu_allocation > 100:
        print("[MPS] Warning, got GPU allocation > 100, capping at 100")
        gpu_allocation = 100
    print("Setting CUDA MPS alloc to {}".format(gpu_allocation))
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(gpu_allocation)