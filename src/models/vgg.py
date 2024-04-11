import torchvision.models as models
import torch.nn as nn

class VGG:
    def __init__(self, version='vgg16', num_classes=10):
        self.version = version
        self.num_classes = num_classes
        self.model = self._construct_model()

    def _construct_model(self):
        if self.version == 'vgg11':
            model = models.vgg11(pretrained=True)
        elif self.version == 'vgg13':
            model = models.vgg13(pretrained=True)
        elif self.version == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif self.version == 'vgg19':
            model = models.vgg19(pretrained=True)
        else:
            raise ValueError(f'Unsupported VGG version: {self.version}')

        # Replace the last layer to match the number of classes
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, self.num_classes)

        return model

    def get_model(self):
        return self.model
