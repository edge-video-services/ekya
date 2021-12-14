# Reference https://github.com/jcjohnson/cnn-benchmarks
# INFER_MAX_RES_DICT = {'resnet18': 0.25 * 31.54/31.54,
#                       'resnet50': 0.25 * 103.58/31.54,
#                       'resnet101': 0.25 * 156.44/31.54}
INFER_MAX_RES_DICT = {'resnet18': 0.25,
                      'resnet50': 0.25,
                      'resnet101': 0.25}
INFINITY = 9999999
INSTA_CHECKPOINT = False
TRAINING_COMPLETE_OVERHEAD = 0  # 4 seconds and this is only an empirical value

# OPT = False  # flag to turn on simulator optimization
OPT = True
