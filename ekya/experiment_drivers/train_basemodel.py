import json
import os

import torch
from torchvision import transforms

from ekya.drivers.motivation.parser import get_parser
from ekya.models.resnet import Resnet
from ekya.datasets.CityscapesClassification import CityscapesClassification
from ekya.datasets.WaymoClassification import WaymoClassification
from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification
from torch.utils.data import random_split

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    NUM_CLASSES = args["num_classes"]
    BATCH_SIZE = 64 # args["batch_size"]
    NUM_EPOCHS = args["epochs"]
    LR = args["learning_rate"]
    MOMENTUM = args["momentum"]
    SAVE_PATH = args["checkpoint_path"]
    root = args["root"]  # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    train_sample_names = args["lists_train"].split(',')
    val_sample_names = args["lists_val"].split(',')
    cache = True

    num_hidden = args["num_hidden"]
    last_layer_only = not args["disable_last_layer_only"]
    model_name = args["model_name"]

    hyperparams = {
        'num_hidden': num_hidden,
        'last_layer_only': last_layer_only,
        'model_name': model_name
    }

    model = Resnet(NUM_CLASSES, hyperparameters=hyperparams)
    trsf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    # train_set = Mp4VideoClassification(
    #     root, train_sample_names, sample_list_root, transform=trsf,
    #     resize_res=224, use_cache=cache)
    # val_set = Mp4VideoClassification(
    #     root, val_sample_names, sample_list_root, transform=trsf,
    #     resize_res=224, use_cache=cache)
    dataset = Mp4VideoClassification(
        root, train_sample_names, sample_list_root, transform=trsf,
        resize_res=224, use_cache=cache)
    train_set, val_set = random_split(dataset, [round(len(dataset)*0.75), round(len(dataset)*0.25)])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args['num_workers'])

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args['num_workers'])

    dataloaders_dict = {'train': train_loader,
                        'val': val_loader}

    _, val_acc_history, best_acc, profile, _ = model.train_model(
        dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR, momentum=MOMENTUM)
    model.save(SAVE_PATH)

    meta = {'args': args,
            'hyperparams': hyperparams,
            'best_acc': best_acc,
            'profile': profile}

    META_PATH = SAVE_PATH + '.meta'
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)

    print("Training complete. Meta: {}".format(meta))
