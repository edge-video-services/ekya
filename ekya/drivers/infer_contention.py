import time

import ray
from ray.experimental import signal

from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Logger

#TODO - This example is incompelete
from ekya.utils.monitoractor import Monitor

def get_model(idx):
    return ray.remote(num_gpus=0.01, num_cpus=0.01)(ICaRLModel).remote(id=idx, logger_actor=logger_actor)

if __name__ == '__main__':
    ray.init()
    init_tasks = []
    checkpoint_restore_times = []
    checkpoint_write_time = 0
    id = 0
    logger_actor = ray.remote(Logger).remote()
    monitor_actor = ray.remote(Monitor).remote(node_id="node", logger_actor=logger_actor)

    for _ in range(0, 30):
        id+=1
        print("Attempting to launch model id {}".format(id))
        logger_actor.append.remote("model_count", [time.time(), id])
        logger_actor.flush.remote()
        infer_model = get_model(id)
        print("model id {}: Restoring model from checkpoint".format(id))
        ray.get(infer_model.restore.remote(path="/tmp/icarl_base.pt"))
        print("model id {}: Model restored, now launching loop".format(id))
        infer_model.infer_loop.remote(0,10)
        time.sleep(30)
        print("model id {}: Sleep done, waiting for signal.".format(id))
        signals = signal.receive([infer_model], timeout=1)
        if not signals:
            print("Launch failed.")
            logger_actor.append.remote("signal_fail_time", [time.time()])
            break
        assert len(signals) == 1
        print("signals {}".format(signals))
        print("signals[0][1].model_id {}".format(signals[0][1].model_id))
        print("Id {}".format(id))
        assert int(signals[0][1].model_id) == id
        print("Launch of model id {} successful.".format(id))