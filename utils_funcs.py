import torch
from collections import OrderedDict
import random
import numpy as np
import os
import logging.config
import matplotlib.pyplot as plt
import random
import yaml
from tqdm import tqdm
import re
import logging
from torch.nn.parameter import Parameter
# original saved file with DataParallel

# create class that scales up the data to [0,255] if called
class ToInt:
    def __call__(self, pic):
        # return pic * 255
        return pic * 511.0


class ToFloat:
    def __call__(self, code):
        return code.to(torch.float32)

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_yaml_(args):
    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

# dynamic adjust values of args
def load_temp_(args):
    def list2dict(l):
        d = {}
        for arg in l:
            k, v = arg.split('=')

            # if '\'' in v:
            #     real_v = v.split('\'')[1]
            #     v = str(real_v)
            d[k] = v
        return d

    if args.temporary is not None:
        d = list2dict(args.temporary)
        for k, v in d.items():
            type_v = type(vars(args)[k])
            if type_v == int:
                vars(args)[k] = int(v)
            elif type_v == float:
                vars(args)[k] = float(v)
            elif type_v == bool:
                vars(args)[k] = bool(v)
            elif type_v == str:
                vars(args)[k] = str(v)


def config_logging(log_file='imgnet32_log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_state_from_ddp_(model, state_dict):
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k[:7]:
            # remove `module.`
            name = k[7:]
            # print('loading {}'.format(name))
        elif 'backbone' in k:
            name = k.replace('backbone.', '')
        else:
            name = k
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)



def reset_last_layer_(model, pretrained_classes, num_classes):
    print('reset last layer')

    in_features = model.fc.in_features
    weight = Parameter(torch.empty((num_classes, in_features)))
    bias = Parameter(torch.empty(num_classes))

    torch.nn.init.kaiming_normal(weight)
    torch.nn.init.constant(bias, 0)
    weight.data[:pretrained_classes, :] = model.fc.weight.data
    bias.data[:pretrained_classes] = model.fc.bias.data
    model.fc.weight = weight
    model.fc.bias = bias


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum += val.cpu() * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n