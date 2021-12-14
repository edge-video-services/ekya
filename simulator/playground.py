import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Model(object):
    def __init__(self, train_acc_vs_t_function, init_train_duration):
        self.train_fn = train_acc_vs_t_function
        self.trained_duration = init_train_duration
        self.init_acc = train_acc_vs_t_function(init_train_duration)

    def post_train_acc(self, train_time):
        # self.trained_duration += train_time # Not updating this yet
        return self.train_fn(train_time + self.trained_duration)

def slowed_acc(acc, contention_slowdown = 0.9):
    return acc*contention_slowdown

def optimus_fn(t, T):
    if t==0:
        return 0
    else:
        return 1/((1/(t*10*1/T)) + 1) * 100   # Hits 0.9 acc at time T. *100 for accuracy.

def inv_optimus_fn(a, T):   # Maps accuract to time
    if a==0:
        return 0
    else:
        return a/(10/T * (100-a))

def linear_fn(t, k):
    return k*t

def inv_linear_fn(a, k):
    return a/k

def get_linear_fn(k):
    return lambda t: linear_fn(t, k), lambda a: inv_linear_fn(a, k)

def get_optimus_fn(T):
    return lambda t: optimus_fn(t, T), lambda a: inv_optimus_fn(a, T)

def get_AUC(train_models, first_duration, second_duration, T):
    if first_duration + second_duration >= T:
        return 0
    # Train_models is a list of models in the order they are trained
    assert len(train_models) == 2
    first_model = train_models[0]
    second_model = train_models[1]
    AUC_first = slowed_acc(first_model.init_acc) * first_duration + slowed_acc(first_model.post_train_acc(first_duration)) * second_duration + first_model.post_train_acc(first_duration) * (T-(first_duration + second_duration))
    AUC_second = slowed_acc(second_model.init_acc) * (first_duration + second_duration) + second_model.post_train_acc(second_duration) * (T-(first_duration + second_duration))
    return AUC_first + AUC_second

def get_AUC_curves(train_models, first_duration, second_duration, T):
    assert first_duration + second_duration <= T
    # Train_models is a list of models in the order they are trained
    assert len(train_models) == 2
    first_model = train_models[0]
    second_model = train_models[1]
    x_data = list(range(0, T))
    acc1 = []   # First accuracy curve
    acc2 = []
    for t in x_data:
        if t < first_duration:
            # Model 1 is training
            acc1.append(slowed_acc(first_model.init_acc))
            acc2.append(slowed_acc(second_model.init_acc))
        elif t >= first_duration and t < first_duration+second_duration:
            # Model 2 is training
            acc1.append(slowed_acc(first_model.post_train_acc(first_duration)))
            acc2.append(slowed_acc(second_model.init_acc))
        elif t >= first_duration + second_duration:
            # Both training done
            acc1.append(first_model.post_train_acc(first_duration))
            acc2.append(second_model.post_train_acc(second_duration))
    print(acc1, acc2, x_data)
    assert len(acc1) == len(acc2) == len(x_data)
    return acc1, acc2, x_data

def plot_acc_data(acc1, acc2, t_data):
    print(acc1, acc2, t_data)
    plt.plot(t_data, acc1, color='r', label='FirstModel')
    plt.plot(t_data, acc2, color='g', label='SecondModel')
    plt.legend()
    plt.title("Accuracy over time")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.show()

def plot_auc_plane(train_models, T):
    A,B = train_models
    train_times = list(range(0, T))
    first_train_time = []
    second_train_time = []
    auc_data = []
    for x in train_times:
        for y in range(0, T - x):
            first_train_time.append(x)
            second_train_time.append(y)
            auc_data.append(get_AUC([A, B], x, y, T))

    max_index = auc_data.index(max(auc_data))
    t1_opt, t2_opt, auc_opt = first_train_time[max_index], second_train_time[max_index], auc_data[max_index]
    print("Max coords: A: {}, B: {}, AUC: {}".format(t1_opt, t2_opt, auc_opt))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.scatter(first_train_time, second_train_time, auc_data)  # , linewidth=0, antialiased=False)
    ax.set_xlabel("A train time")
    ax.set_ylabel("B train time")
    ax.set_zlabel("AUC")
    ax.set_title("AUC vs A train time vs B train time")

    plt.figure()
    acc1, acc2, t_data = get_AUC_curves([A, B], t1_opt, t2_opt, T)
    plot_acc_data(acc1, acc2, t_data)
    plt.show()

a_conv_time = 1
b_conv_time = 10
target_start_accuracy = 10
a_func, a_inv_func = get_optimus_fn(a_conv_time)
b_func, b_inv_func = get_optimus_fn(b_conv_time)
init_time_a = a_inv_func(target_start_accuracy)
init_time_b = b_inv_func(target_start_accuracy)
A = Model(a_func, init_time_a)
B = Model(b_func, init_time_b)

plot_auc_plane([A,B], T=20)