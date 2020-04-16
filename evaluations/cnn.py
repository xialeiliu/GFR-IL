from collections import OrderedDict

from torch.autograd import Variable
from utils import to_torch
import torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs).cuda()
    with torch.no_grad():
        if modules is None:
            outputs = model(inputs)
            outputs = outputs.data.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None

            def func(m, i, o): outputs[id(m)] = o.data.cpu()

            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
    return list(outputs.values())


def extract_cnn_feature_classification(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs).cuda()
    with torch.no_grad():
        if modules is None:
            outputs = model.extract_feat(inputs)
            outputs = outputs.data.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None

            def func(m, i, o): outputs[id(m)] = o.data.cpu()

            handles.append(m.register_forward_hook(func))
        model.extract_feat(inputs)
        for h in handles:
            h.remove()
    return list(outputs.values())
