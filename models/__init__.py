from .resnet import resnet101
from .resnet import resnet50
from .resnet import resnet18
from .resnet import resnet18_cifar
from .resnet import resnet18_imagenet

__factory = {
    #'resnet18': resnet18,
    'resnet18_cifar': resnet18_cifar,
    'resnet18_imagenet': resnet18_imagenet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)

