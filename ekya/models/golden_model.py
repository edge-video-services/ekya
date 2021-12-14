"""Abstraction of golden model in ekya."""
import torch
import torchvision.models as models

from ekya.models.resnext import (resnext50, resnext50_elastic, se_resnext50,
                                 se_resnext50_elastic, resnext101,
                                 resnext101_elastic)
COCO_MODELS = {'resnext50', 'resnext50_elastic', 'resnext101',
               'resnext101_elastic'}


def get_model_from_str(model_str):
    if model_str == 'resnet18':
        return models.resnet18
    if model_str == 'resnet50':
        return models.resnet50
    if model_str == 'resnet101':
        return models.resnet101
    if model_str == 'resnet152':
        return models.resnet152
    if model_str == 'resnext50':
        return resnext50
    if model_str == 'resnext50_elastic':
        return resnext50_elastic
    if model_str == 'se_resnext50':
        return se_resnext50
    if model_str == 'se_resnext50_elastic':
        return se_resnext50_elastic
    if model_str == 'resnext101':
        return resnext101
    if model_str == 'resnext101_elastic':
        return resnext101_elastic

    raise Exception("Model {} not found".format(model_str))


class GoldenModel(object):
    """Golden image classification model of ekya.

    Assume golden model is pretrained on COCO.
    """

    def __init__(self, model_name, checkpoint_path, target_dataset,
                 device=None):
        if device is not None and torch.cuda.is_available():
            if device < 0 or device >= torch.cuda.device_count():
                raise RuntimeError(f'Invalid device number {device}.')
            else:
                self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')
        # Torchvision model trained on imagenet
        self.model_class = get_model_from_str(model_name)
        if model_name in COCO_MODELS:
            self.pretrained_dataset = 'coco'
            num_classes = 80
            print(f'load {model_name}')
            self.model = self.model_class(num_classes=num_classes)
            self.model = torch.nn.DataParallel(
                self.model, device_ids=[self.device])
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if ('module.fc.bias' in checkpoint['state_dict'] and
                checkpoint['state_dict']['module.fc.bias'].size() ==
                self.model.module.fc.bias.size()) or \
                ('module.classifier.bias' in checkpoint['state_dict'] and
                 checkpoint['state_dict']['module.classifier.bias'].size() ==
                 self.model.module.classifier.bias.size()):
                self.model.load_state_dict(
                    checkpoint['state_dict'], strict=False)
            else:
                raise RuntimeError(
                    f'{model_name} module and checkpoint do not match.')
        # else:
        #     self.model = self.model_class(pretrained=True)
        print(f"Golden Model({model_name}) is loaded to device cuda.")
        self.target_dataset = target_dataset

    def infer(self, dataloader):
        pred_labels = []
        self.model.eval()
        running_corrects = 0

        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                preds += 1
                preds_copy = preds.clone()
                #  depend on training dataset need to perform label mapping
                if self.target_dataset == 'waymo':
                    # 'vehicle': 0
                    # 'pedestrian': 1
                    # 'sign': 2
                    # 'cyclist': 3
                    preds[preds_copy == 3] = 0  # car
                    preds[preds_copy == 4] = 0  # motorcycle
                    preds[preds_copy == 6] = 0  # bus
                    preds[preds_copy == 8] = 0  # truck
                    preds[preds_copy == 1] = 1  # person
                    preds[preds_copy == 10] = 2  # traffic light
                    preds[preds_copy == 12] = 2  # street sign
                    preds[preds_copy == 13] = 2  # stop sign
                    preds[preds_copy == 14] = 2  # parking meter
                    preds[preds_copy == 2] = 3  # bicyle
                elif self.target_dataset == 'cityscapes':
                    # 'person': 0
                    # 'car': 1
                    # 'truck': 2
                    # 'bus': 3
                    # 'bicycle': 4
                    # 'motorcycle': 5
                    preds[preds_copy == 1] = 0  # person
                    preds[preds_copy == 3] = 1  # car
                    preds[preds_copy == 8] = 2  # truck
                    preds[preds_copy == 6] = 3  # bus
                    preds[preds_copy == 2] = 4  # bicyle
                    preds[preds_copy == 4] = 5  # motorcycle

                    preds[preds_copy == 5] = 81  # remove airplane
            pred_labels.extend(preds.data.tolist())

            num_corrects = int(torch.sum(preds == labels.data).cpu().data)
            # if num_corrects != len(preds):
            #     print('label:', labels)
            #     print('preds:', preds)
            #     print('probs:', outputs)
            running_corrects += num_corrects

            print('Infer [{}/{} ({:.0f}%)]\tBatch acc: {:.2f}% \t'
                  'Running acc: {:.2f}%'.format(
                      batch_idx * len(inputs), len(dataloader) * len(inputs),
                      100. * batch_idx / len(dataloader),
                      num_corrects / len(inputs),
                      running_corrects / ((batch_idx+1) * len(inputs))))

        acc = running_corrects / len(dataloader.dataset)

        print('Inference done. Final Acc: {:.4f}'.format(acc))
        return float(acc), pred_labels
