
import torch
from torch import nn

from resnext_weights.resnext import resnext

def get_ResNeXt(num_classes=10, pretrained=True, include_classifier=False, **kwargs):
    '''
    Load pretrained ResNeXt model trained on CIFAR100, replace trained classifier with untrained classifier, before using download weights from https://github.com/bearpaw/pytorch-classification
    '''
    model = resnext(
    cardinality=8,
    num_classes=100,
    depth=29,
    widen_factor=4,
    dropRate=0,
    )

    # load pretrained weights
    if pretrained:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load("resnext_weights/resnext_8x64d/model_best.pth.tar", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model = list(model.children())[0]
        for param in model.parameters():
            param.requires_grad = False
        if include_classifier:
            model.classifier = nn.Linear(1024, num_classes, bias=True)
            list(model.parameters())[-1].requires_grad = True
        else:
            model.classifier = nn.Identity()
    return model

class ResNeXt(nn.Module):
    """Wrapper for pretrained resnext model."""
    def __init__(self, input_shapes, num_classes:int=10, pretrained:bool=True, **kwargs):
        super(ResNeXt, self).__init__()
        self.input_shapes = input_shapes
        self.num_classes = num_classes
        self.model = get_ResNeXt(num_classes=num_classes, pretrained=pretrained, **kwargs)
        self.classifier = self.model.classifier

    def forward(self, x):
        x = x.view(-1, *self.input_shapes)
        return self.model(x)

# CNN structure for MNIST/Fashion MNIST from https://arxiv.org/abs/2011.11660
class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# from https://arxiv.org/abs/2011.11660
def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """
    def __init__(self, num_inp, num_out, bias):
        super(LinearClassificationNet, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_out, bias=bias)
    
    def forward(self, x):
        x = self.layer1(x)
        return x
