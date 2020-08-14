from src.train import Trainer
from src.datasets import IRMAS
from src.utils import get_network
from torch import nn
from torch.utils import data as tdata

import yaml
class ParamsCont:
    
    def __init__(self, data):
        
        for key, val in data.items():
            if isinstance(val, dict):
                self.__dict__[key] = ParamsCont(val)
            else: 
                self.__dict__[key] = val
    
    @staticmethod    
    def from_path(_cls, path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        obj = _cls(data)
        return obj

p = ParamsCont.from_path('experiments/template.yaml')

# trans = None
# trn_ds = IRMAS(mode='train', is_test=True, transforms=trans)
# val_ds = IRMAS(mode='val', is_test=True, transforms=trans)
# trn_loader = tdata.DataLoader(
#     trn_ds, batch_size=32, num_workers=8)
# val_loader = tdata.DataLoader(
#     val_ds, batch_size=32, num_workers=8)
    
# model = get_network('resnet18', False, True)
# criterion = nn.BCEWithLogitsLoss()
# root = 'data/results/test'
# trainer = Trainer(root, model, trn_loader, val_loader, criterion)

# trainer.train(2)