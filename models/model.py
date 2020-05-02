from pretrainedmodels import models as pm
import pretrainedmodels
from torch import nn
from torchvision import models as tm
from config import configs
from efficientnet_pytorch import EfficientNet
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.parameter import Parameter

weights = {
        "efficientnet-b3":"/data/dataset/detection/pretrainedmodels/efficientnet-b3-c8376fa2.pth",
        "efficientnet-b4":"/data/dataset/detection/pretrainedmodels/efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5":"/data/dataset/detection/pretrainedmodels/efficientnet-b5-b6417697.pth",
        "efficientnet-b6":"/data/dataset/detection/pretrainedmodels/efficientnet-b6-c76e70fd.pth",
        }

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def get_model():
    if configs.model_name.startswith("resnext50_32x4d"):
        model = tm.resnext50_32x4d(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048,configs.num_classes)
        model.cuda()
    elif configs.model_name.startswith("efficient"):
        # efficientNet
        model_name = configs.model_name[:15]
        model = EfficientNet.from_name(model_name)
        model.load_state_dict(torch.load(weights[model_name]))
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
                        nn.BatchNorm1d(in_features),
                        nn.Dropout(0.5),
                        nn.Linear(in_features, configs.num_classes),
                         )
        model.cuda()
    else:
        pretrained = "imagenet+5k" if configs.model_name.startswith("dpn") else "imagenet"
        model = pretrainedmodels.__dict__[configs.model_name.split("-model")[0]](num_classes=1000, pretrained=pretrained)
        if configs.model_name.startswith("pnasnet"):
            model.last_linear = nn.Linear(4320, configs.num_classes)
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif configs.model_name.startswith("inception"):
            model.last_linear = nn.Linear(1536, configs.num_classes)
            model.avgpool_1a  = nn.AdaptiveAvgPool2d(1)            
        else:
            model.last_linear = nn.Linear(2048, configs.num_classes)
            model.avg_pool = nn.AdaptiveAvgPool2d(1)           
        
        model.cuda()
    return model