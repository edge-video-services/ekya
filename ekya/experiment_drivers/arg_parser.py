import argparse

def get_parser():
    parser = argparse.ArgumentParser("Ekya scripts",
                                     description="Ekya script.")

    # Core
    parser.add_argument("-ld", "--log-dir", default='/tmp/ekya', type=str,
                        help="Directory to log results to")
    parser.add_argument("-retp", "--retraining-period", default=180, type=int,
                        help="Retraining period in seconds")
    parser.add_argument("-infc", "--inference-chunks", default=10, type=int,
                        help="Number of inference chunks per retraining window.")
    parser.add_argument("-numgpus", "--num-gpus", default=1, type=int,
                        help="Number of GPUs to partition.")
    parser.add_argument("-memgpu", "--gpu-memory", default=0, type=int,
                        help="Per GPU Memory in GB.")

    # Data related:
    default_root = '/home/romilb/datasets/cityscapes_raw/'
    parser.add_argument("-r", "--root", default=default_root, type=str,
                        help="Cityscapes dataset root.")
    parser.add_argument("--dataset-name", default="cityscapes", type=str,
                        choices=['cityscapes', 'waymo', 'vegas', 'bellevue'],
                        help="Name of the dataset supported.")
    parser.add_argument("-c", "--cities", default="aachen", type=str,
                        help="comma separated str of list of cities to create cameras. Num cameras = num of cities")
    parser.add_argument("-lpt", "--lists-pretrained", default="", type=str,
                        help="comma separated str of lists used for training the pretrained model. Used as history for continuing the retraining. Usually frankfurt,munster.")
    parser.add_argument("-lp", "--lists-root", default="sample_lists/citywise/", type=str,
                        help="root of samplelists")
    parser.add_argument("-dc", "--use-data-cache", action="store_true", default=True,
                        help="Use data caching for cityscapes. WARNING: Might consume lot of disk space.")
    parser.add_argument("-ir", "--resize-res", default=224, type=int,
                        help="Image size to use for cityscapes.")
    parser.add_argument("-w", "--num-workers", default=0, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-ts", "--train-split", default=0.8, type=float,
                        help="Train validation split. This float is the fraction of data used for training, rest goes to validation.")
    parser.add_argument("--golden-label", action="store_true", default=False,
                        help="Use golden model labels as ground truth if "
                        "specified. Otherwise, use human label groundtruth. "
                        "Default: False")


    # Training related:
    parser.add_argument("-dtfs", "--do-not-train-from-scratch", action="store_false", default=True,
                        help="Do not train from scratch for every profiling task - carry forward the previous model")
    parser.add_argument("-hw", "--history-weight", default=-1, type=float,
                        help="Weight to assign to historical samples when retraining. Between 0-1. Cannot be zero. -1 if no reweighting.")
    parser.add_argument("-rp", "--restore-path", default='/home/romilb/research/msr/models', type=str,
                        help="Path from where to restore the model for initialization")
    parser.add_argument("-cp", "--checkpoint-path", default='', type=str,
                        help="Path where to save the model")
    parser.add_argument("-mn", "--model-name", default="resnet18", type=str,
                        help="Model name. Can be resnetXX for now.")
    parser.add_argument("-nc", "--num-classes", default=6, type=int,
                        help="Number of classes per task.")
    parser.add_argument("-b", "--batch-size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
                        help="Learning rate.")
    parser.add_argument("-mom", "--momentum", default=0.9, type=float,
                        help="Momentum.")
    parser.add_argument("-e", "--epochs", default=30, type=int,
                        help="Number of epochs per task.")
    parser.add_argument("-nh", "--num-hidden", default=512, type=int,
                        help="Number of neurons in hidden layer.")
    parser.add_argument("-dllo", "--disable-last-layer-only", action="store_true", default=False,
                        help="Adjust weights on all layers, instead of modifying just last layer.")
    # parser.add_argument("-wd", "--weight-decay", default=0.00001, type=float,
    #                     help="Weight decay.")
    # parser.add_argument("-sc", "--scheduling", default=[50, 64], nargs="*", type=int,
    #                     help="Epoch step where to reduce the learning rate.")
    # parser.add_argument("-lr-decay", "--lr-decay", default=1/5, type=float,
    #                     help="LR multiplied by it.")
    # parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
    #                     help="Optimizer to use.")
    parser.add_argument("--golden-model-ckpt-path", type=str, default="",
                        help='Path from where to load the golden model weights.')

    # Scheduler related:
    parser.add_argument("-sched", "--scheduler", default="noretrain", type=str,
                        help="Scheduler to use. Either of fair, noretrain, thief, utilitysim.")

    # Utility sim scheduler args
    parser.add_argument("-usp", "--utilitysim-schedule-path", default="", type=str,
                        help="Path to the schedule (period allocation) generated by utilitysim.")
    # parser.add_argument("-uhp", "--utilitysim-hyps-path", default="", type=str,
    #                     help="hyp_map.json path which lists the hyperparameter_id to hyperparameter mapping.")
    parser.add_argument("-usc", "--utilitysim-schedule-key", default="100_1_thief_True", type=str,
                        help="The top level key in the schedule json. Usually of the format {}_{}_{}_{}.format(period,res_count,scheduler,use_oracle)")

    # Thief scheduler args
    parser.add_argument("-mpd", "--microprofile-device", default="cuda", type=str,
                        help="Device to microprofile on - either of cuda, cpu or auto")
    parser.add_argument("-mprpt", "--microprofile-resources-per-trial", default=0.5, type=float,
                        help="Resources required per trial in microprofiling. Reduce this to run multiple jobs in together while microprofiling. Warning: may cause OOM error if too many run together.")
    parser.add_argument("-mpe", "--microprofile-epochs", default=5, type=int,
                        help="Epochs to run microprofiling for.")
    parser.add_argument("-mpsr", "--microprofile-subsample-rate", default=0.1, type=float,
                        help="Subsampling rate while microprofiling.")
    parser.add_argument("-mpep", "--microprofile-profiling-epochs", default="5, 15, 30", type=str,
                        help="Epochs to generate profiles for, per hyperparameter.")

    # Fair scheduler args
    parser.add_argument("-fswt", "--fair-inference-weight", default=0.5, type=float,
                        help="Weight to allocate for inference in the fair scheduler.")

    # Misc:
    parser.add_argument("-nt", "--num-tasks", default=10, type=int,
                        help="Number of tasks to split each dataset into")
    parser.add_argument("-stid", "--start-task", default=0, type=int,
                        help="Task id to start at.")
    parser.add_argument("-ttid", "--termination-task", default=-1, type=int,
                        help="Task id to end the Ekya loop at. -1 runs all tasks.")
    parser.add_argument("-nsp", "--num-subprofiles", default=3, type=int,
                        help="Number of tasks to split each dataset into")
    parser.add_argument("-op", "--results-path", default='results.json', type=str,
                        help="The josn file to write results to.")
    parser.add_argument("-uhp", "--hyps-path", default="", type=str,
                        help="hyp_map.json path which lists the hyperparameter_id to hyperparameter mapping.")
    parser.add_argument("-hpid", "--hyperparameter-id", default="0", type=str,
                        help="Hyperparameter id to use for retraining. From hyps-path json.")

    # Profiling:
    parser.add_argument("-pm", "--profiling-mode", action="store_true", default=False,
                        help="Run in profiling mode?")
    parser.add_argument("-pp", "--profile-write-path", default="0", type=str,
                        help="Run in profiling mode?")

    # Inference profiling args
    parser.add_argument("-ipp", "--inference-profile-path", default='real_inference_profiles.csv', type=str,
                        help="Path to the inference profiles csv")
    parser.add_argument("-mir", "--max-inference-resources", default=0.25, type=float,
                        help="Maximum resources required for inference. Acts as a ceiling for the inference scaling function.")
    # parser.add_argument("--device", default=0, type=int,
    #                     help="GPU index to use, for cpu use -1.")
    # parser.add_argument("-seed", "--seed", default=[1], type=int, nargs="+",
    #                     help="Random seed.")

    return parser

def get_defaults_dict():
    parser_instance = get_parser()
    args = parser_instance.parse_args(args=[])
    all_defaults = {}
    for key in vars(args):
        all_defaults[key] = parser_instance.get_default(key)
    return all_defaults

def fill_in_defaults(args):
    defaults = get_defaults_dict()
    for k,v in defaults.items():
        if k not in args:
            args[k] = v
    return args
