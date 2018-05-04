import torch.nn as nn
import torchvision.transforms as transforms
from .mean_bn import MeanBN
__all__ = ['vgg_cifar100']

class VggCifra100(nn.Module):

    def __init__(self, num_classes=100):
        super(VggCifra100, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            MeanBN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MeanBN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                      bias=False),
            MeanBN(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            MeanBN(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            MeanBN(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            MeanBN(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            MeanBN(256),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            MeanBN(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            MeanBN(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            MeanBN(512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2 , 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #nn.Linear(1024, 1024, bias=False),
            #BatchNorm1d_big(1024),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 5.7e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            32*50: {'lr': 5.7*5e-3},
            32*80: {'lr': 5.7*1e-3, 'weight_decay': 0},
            32*110: {'lr': 5.7*5e-4},
            32*120: {'lr': 5.7*1e-4},
            32*130: {'lr': 5.7*1e-5}
        }
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 5.7e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    50: {'lr': 5.7*5e-3},
        #    80: {'lr': 5.7*1e-3, 'weight_decay': 0},
        #    110: {'lr': 5.7*5e-4},
        #    120: {'lr': 5.7*1e-4},
        #    130: {'lr': 5.7*1e-5}
        #}

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 2 * 2)
        x = self.classifier(x)
        return x


def vgg_cifar100(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 100)
    return VggCifra100(num_classes)
