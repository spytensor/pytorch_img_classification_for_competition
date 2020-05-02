import os
import torch
import shutil
import pandas as pd
from .optimizers import *
from config import configs
from torch import optim as optim_t
from tqdm import tqdm
from glob import glob
from itertools import chain

def get_optimizer(model):
    if configs.optim == "adam":
        return optim_t.Adam(model.parameters(),
                            configs.lr,
                            betas=(configs.beta1,configs.beta2),
                            weight_decay=configs.wd)
    elif configs.optim == "radam":
        return RAdam(model.parameters(),
                    configs.lr,
                    betas=(configs.beta1,configs.beta2),
                    weight_decay=configs.wd)
    elif configs.optim == "ranger":
        return Ranger(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "over9000":
        return Over9000(model.parameters(),
                        lr = configs.lr,
                        betas=(configs.beta1,configs.beta2),
                        weight_decay=configs.wd)
    elif configs.optim == "ralamb":
        return Ralamb(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "sgd":
        return optim_t.SGD(model.parameters(),
                        lr = configs.lr,
                        momentum=configs.mom,
                        weight_decay=configs.wd)
    else:
        print("%s  optimizer will be add later"%configs.optim)

def save_checkpoint(state,is_best,is_best_loss):
    filename = configs.checkpoints + os.sep + configs.model_name + "-checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = filename.replace("-checkpoint.pth.tar","-best_model.pth.tar")
        shutil.copyfile(filename, message)
    if is_best_loss:
        message = filename.replace("-checkpoint.pth.tar","-best_loss.pth.tar")
        shutil.copyfile(filename, message)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_files(root,mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    else:
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split(os.sep)[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = [5e-4, 1e-4, 1e-5, 1e-6]
    if epoch<=10:
        lr = lrs[0]
    elif epoch>10 and epoch<=16:
        lr = lrs[1]
    elif epoch>16 and epoch<=22:
        lr = lrs[2]
    else:
        lr = lrs[-1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr