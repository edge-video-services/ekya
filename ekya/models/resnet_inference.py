import torch
from torchvision.models.resnet import resnet50
from ekya.models.inference_manager import InferenceModel
from torchvision import transforms
from PIL import Image

class ResnetInference(InferenceModel):
    def __init__(self):
        self.model = resnet50(pretrained=False)
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)

    def infer_image(self, img_path):
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        img = Image.open(img_path)
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        return self.model(batch_t)

    def infer_images(self, list_of_images):
        results = []
        for img_path in list_of_images:
            results.append(self.infer_image(img_path))
        return results
