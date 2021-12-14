import json
import os
import random
import time

import numpy as np
import torch


def timedrun(method, *args, **kwargs):
    start_time = time.time()
    result = method(*args, **kwargs)
    end_time = time.time()
    time_taken = end_time-start_time
    return time_taken, result

def seed_all(seed):
    seed_python(seed)
    seed_pytorch(seed)

def seed_python(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seed_pytorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json_file(filename):
    """Load json object from a file."""
    with open(filename, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(filename, content):
    """Dump into a json file."""
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
