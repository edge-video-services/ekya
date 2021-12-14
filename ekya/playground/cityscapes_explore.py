import torch
from torchvision.datasets.cityscapes import Cityscapes

path = '/home/mnr/cityscapes/'
path = '/media/romilb/NEW VOLUME/cityscapes/dataset'
d = Cityscapes(root=path, target_type='polygon')