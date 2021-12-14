import json

from ekya.models.resnet import Resnet
import torch
from torchvision import transforms
import os
from inclearn.lib.CityscapesClassification import CityscapesClassification
from ekya.drivers.motivation.parser import get_parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    NUM_CLASSES = args["num_classes"]
    BATCH_SIZE = args["batch_size"]
    NUM_EPOCHS = args["epochs"]
    LR=args["learning_rate"]
    MOMENTUM=args["momentum"]
    SAVE_PATH=args["checkpoint_path"]
    root = args["root"] # Dataset root
    sample_list_root = os.path.join(root, args["lists_root"])
    train_sample_names = args["lists_train"].split(',')
    val_sample_names = args["lists_val"].split(',')
    cache = args["use_data_cache"]

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
    train_set = CityscapesClassification(root, train_sample_names, sample_list_root, transform=trsf, resize_res=224, use_cache=cache)
    val_set = CityscapesClassification(root, val_sample_names, sample_list_root, transform=trsf, resize_res=224, use_cache=cache)

    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args['num_workers']),
                        'val': torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args['num_workers'])}

    _, val_acc_history, best_acc, profile, _ = model.train_model(dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR, momentum=MOMENTUM)
    model.save(SAVE_PATH)

    meta = {'args': args,
            'hyperparams': hyperparams,
            'best_acc': best_acc,
            'profile': profile}

    META_PATH = SAVE_PATH + '.meta'
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)

    print("Training complete. Meta: {}".format(meta))