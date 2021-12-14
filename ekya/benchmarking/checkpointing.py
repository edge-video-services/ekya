import time

from ekya.models.resnet import Resnet

init_start_time = time.time()
model = Resnet(6, hyperparameters=Resnet.DEFAULT_HYPER_PARAMS)

save_start_time = time.time()
model = Resnet(6, hyperparameters=Resnet.DEFAULT_HYPER_PARAMS)
model.save('/dev/shm/cp.pt')
save_end_time = time.time()
save_time = save_end_time - save_start_time
init_time = save_start_time - init_start_time
print("Save time taken = {}".format(save_time))
print("Cold Init time taken = {}".format(init_time))
print(model.device)

input("Press enter.")
# Restore:
init_start_time = time.time()
model = Resnet(6, hyperparameters=Resnet.DEFAULT_HYPER_PARAMS)
save_start_time = time.time()
model.load('/dev/shm/cp.pt')
save_end_time = time.time()
save_time = save_end_time - save_start_time
init_time = save_start_time - init_start_time
print("Load time taken = {}".format(save_time))
print("Warm  Init time taken = {}".format(init_time))
