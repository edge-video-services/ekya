import os

import pandas as pd

from torchvision import transforms
from torchvision.datasets import VisionDataset

from ekya.CONFIG_DATASET import CITYSCAPES_PATH


def get_dataset(name: str) -> [VisionDataset, dict]:
    '''
    Returns a dataset and the default kwargs to be used with the dataset, if any.
    :param name:
    :return:
    '''
    name = name.lower()
    if name == 'cityscapes':
        from ekya.datasets.CityscapesClassification import CityscapesClassification
        trsf = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        dataset_class = CityscapesClassification
        default_args = {"trsf": trsf,
                        "use_cache": True,
                        "root": CITYSCAPES_PATH,
                        "sample_list_root": os.path.join(CITYSCAPES_PATH, 'sample_lists', 'citywise'),
                        "num_classes": 6}
    elif name == 'waymo':
        from ekya.datasets.WaymoClassification import WaymoClassification
        trsf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        dataset_class = WaymoClassification
        default_args = {"trsf": trsf,
                        "use_cache": True,
                        "num_classes": 4}
    elif name == 'vegas' or name == 'bellevue':
        from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification
        trsf = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        dataset_class = Mp4VideoClassification
        default_args = {"trsf": trsf,
                        "use_cache": True,
                        "num_classes": 6}
    else:
        raise NotImplementedError("Dataset {} not implemented.".format(name))
    return dataset_class, default_args

def has_header(file, nrows=5):
    df = pd.read_csv(file, header=None, nrows=nrows)
    df_header = pd.read_csv(file, nrows=nrows)
    return tuple(df.dtypes) != tuple(df_header.dtypes)

def get_header(file, nrows=5):
    if has_header(file, nrows):
        header=0
    else:
        header=None

def get_pretrained_model_format(dataset_name,
                              pretrained_model_dir):
    if dataset_name == 'cityscapes':
        path_format = "pretrained_cityscapes_fftmunster_{}_{}x2.pt"
    elif dataset_name == 'waymo':
        path_format = "waymo_{}_{}.pth"
    elif dataset_name == 'vegas':
        # use cityscapes pretrained models for now
        path_format = "pretrained_mp4_las_vegas_{}_{}x2.pt"
    elif dataset_name == 'bellevue':
        path_format = "pretrained_bellevue_Bellevue_116th_NE12th__2017-09-10_20-09-12_0.5fps_Bellevue_116th_NE12th__2017-09-10_21-08-54_0.5fps_Bellevue_116th_NE12th__2017-09-10_23-08-29_0.5fps_{}_{}x2.pt"
    else:
        raise NotImplementedError('Unknown dataset.')
    return os.path.join(pretrained_model_dir, path_format)
