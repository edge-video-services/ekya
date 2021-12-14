import argparse


def get_parser():
    parser = argparse.ArgumentParser("Ekya scripts",
                                     description="Ekya script.")

    # Data related:
    default_root = '/home/romilb/datasets/cityscapes_raw/'
    parser.add_argument("-r", "--root", default=default_root, type=str,
                        help="Cityscapes dataset root.")
    parser.add_argument("-lt", "--lists-train", default="aachen,zurich", type=str,
                        help="comma separated str of list of  to use for training")
    parser.add_argument("-lv", "--lists-val", default="bremen", type=str,
                        help="comma separated str of list of  to use for validation")
    parser.add_argument("-lpt", "--lists-pretrained", default="frankfurt,munster", type=str,
                        help="comma separated str of lists used for training the pretrained model. Used as history for continuing the retraining.")
    parser.add_argument("-lp", "--lists-root", default="sample_lists/citywise/", type=str,
                        help="root of samplelists")
    parser.add_argument("-dc", "--use-data-cache", action="store_true", default=False,
                        help="Use data caching for cityscapes. WARNING: Might consume lot of disk space.")
    parser.add_argument("-ir", "--resize-res", default=224, type=int,
                        help="Image size to use for cityscapes.")
    parser.add_argument("-w", "--num-workers", default=0, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-vf", "--validation-frequency", default=1, type=int,
                        help="Run validation every n epochs.")
    parser.add_argument("-ts", "--train-split", default=0.8, type=float,
                        help="Train validation split. This float is the fraction of data used for training, rest goes to validation.")


    # Training related:
    parser.add_argument("-dtfs", "--do-not-train-from-scratch", action="store_false", default=True,
                        help="Do not train from scratch for every profiling task - carry forward the previous model")
    parser.add_argument("-hw", "--history-weight", default=-1, type=float,
                        help="Weight to assign to historical samples when retraining. Between 0-1. Cannot be zero. -1 if no reweighting.")
    parser.add_argument("-rp", "--restore-path", default='', type=str,
                        help="Path from where to restore the model for initialization")
    parser.add_argument("-cp", "--checkpoint-path", default='', type=str,
                        help="Path where to save the model")
    parser.add_argument("-mn", "--model-name", default="resnet18", type=str,
                        help="Model name. Can be resnetXX for now.")
    parser.add_argument("-nc", "--num-classes", default=6, type=int,
                        help="Number of classes per task.")
    parser.add_argument("-ob", "--override-batch-size", default=0, type=int,
                        help="Batch size to override default hyperparams")
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

    # Misc:
    parser.add_argument("-nt", "--num-tasks", default=10, type=int,
                        help="Number of tasks to split each dataset into")
    parser.add_argument("-nsp", "--num-subprofiles", default=3, type=int,
                        help="Number of tasks to split each dataset into")
    parser.add_argument("-op", "--results-path", default='results.json', type=str,
                        help="The josn file to write results to.")
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