import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def make_model(pretrained=True, fixed=False):
    """
    Make a densenet121 model with output of dimension of [batch_size, 41]
    Args:
        - pretrained: whether load the pretrained weights
    Returns:
        - model: the customized densenet121 model    
    """
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(in_features=1024, out_features=41, bias=True)
    
    if fixed:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model