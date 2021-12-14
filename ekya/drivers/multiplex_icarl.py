import time

import ray
from ekya.models.icarl_model import ICaRLModel
from ekya.utils.loggeractor import Loggertrain_baseline_model

#TODO - This example is incompelete
from ekya.utils.monitoractor import Monitor

if __name__ == '__main__':
    ray.init()
    init_tasks = []
    checkpoint_restore_times = []
    checkpoint_write_time = 0
    logger_actor = ray.remote(Logger).remote()
    monitor_actor = ray.remote(Monitor).remote(node_id="node", logger_actor=logger_actor)
    infer_model = ray.remote(num_gpus=0.01)(ICaRLModel).remote(id="1", logger_actor=logger_actor)
    train_model = ray.remote(num_gpus=0.01)(ICaRLModel).remote(id="2", logger_actor=logger_actor)

    init_tasks.append(train_model.restore.remote(path="/tmp/icarl_base.pt"))
    init_tasks.append(infer_model.set_handle.remote(infer_model))
    init_tasks.append(infer_model.restore.remote(path="/tmp/icarl_base.pt"))
    ray.get(init_tasks)

    # Run inference on the first task
    for i in range(0,10):
        infer_result = ray.get(infer_model.infer.remote(0, high_class_idx=10))
    print("Task 1 Inference Result: {}".format(infer_result))

    # Run inference on the second task - unknown classes and thus accuracy suffers.

    for i in range(0,3):
        infer_result = ray.get(infer_model.infer.remote(0, high_class_idx=20))
    # TODO: Get accuracy and save it using logger
    print("Task 2 Inference Result: {}".format(infer_result))

    # Start retrain on the second task
    train_future = train_model.train.remote(low_class_idx=10, high_class_idx=20, n_epochs=50, checkpoint_interval=10)

    for _ in range(0,20):
        time.sleep(15)
        ray.get(infer_model.restore.remote(path="/tmp/icarl_updated.pt"))
        infer_result = ray.get(infer_model.infer.remote(0, high_class_idx=20))
        print("Coexisting Inference Result: {}".format(infer_result))

    # Checkpoint new model once retrain completes
    ray.get(train_future)
    start_time = time.time()
    ray.get(train_model.checkpoint.remote(path="/tmp/icarl_updated.pt"))
    checkpoint_write_time = time.time() - start_time

    # Update inference model with the new checkpoint
    start_time = time.time()
    ray.get(infer_model.restore.remote(path="/tmp/icarl_updated.pt"))
    checkpoint_restore_times.append(time.time()-start_time)

    # Run inference on the second task - accuracy improves again.
    infer_result = ray.get(infer_model.infer.remote(0, high_class_idx=20))
    # TODO: Get accuracy and save it using logger
    print("Task 5 Inference Result: {}".format(infer_result))

    print("Checkpoint write times: {}".format(checkpoint_write_time))
    print("Checkpoint restore times: {}".format(checkpoint_restore_times))

# Exemplar time = 14.5s - ADD TO CHECKPOINT WRITE TIME
# Checkpoint write times: 0.7461111545562744
# Checkpoint restore times: [0.3464539051055908]