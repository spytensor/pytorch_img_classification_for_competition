import random
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import ImageFile
from models.model import get_model
from sklearn.model_selection import train_test_split
from utils.misc import *
from utils.logger import *
from utils.losses import *
from progress.bar import Bar
from utils.reader import WeatherDataset

# for train fp16
if configs.fp16:
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id

# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(configs.seed)

# make dir for use
def makdir():
    if not os.path.exists(configs.checkpoints):
        os.makedirs(configs.checkpoints)
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    if not os.path.exists(configs.submits):
        os.makedirs(configs.submits)
makdir()

best_acc = 0  # best test accuracy
best_loss = 999 # lower loss

def main():
    global best_acc
    global best_loss
    start_epoch = configs.start_epoch
    # set normalize configs for imagenet
    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(configs.input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize_imgnet
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(int(configs.input_size * 1.2)),
        transforms.CenterCrop(configs.input_size),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    # Data loading code
    if configs.split_online:
        # use online random split dataset method
        total_files = get_files(configs.dataset,"train")
        train_files,val_files = train_test_split(total_files,test_size = 0.1,stratify=total_files["label"])
        train_dataset = WeatherDataset(train_files,transform_train)
        val_dataset = WeatherDataset(val_files,transform_val)
    else:
        # use offline split dataset
        train_files = get_files(configs.dataset+"/train/","train")
        val_files = get_files(configs.dataset+"/val/","train")
        train_dataset = WeatherDataset(train_files,transform_train)
        val_dataset = WeatherDataset(val_files,transform_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.bs, shuffle=True,
        num_workers=configs.workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.bs, shuffle=False,
        num_workers=configs.workers, pin_memory=True
    )    
    # get model
    model = get_model()
    model.cuda()
    # choose loss func,default is CE
    if configs.loss_func == "LabelSmoothCE":
        criterion = LabelSmoothingLoss(0.1, configs.num_classes).cuda()
    elif configs.loss_func == "CrossEntropy":
        criterion = nn.CrossEntropyLoss().cuda()
    elif configs.loss_func == "FocalLoss":
        criterion = FocalLoss(gamma=2).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model)
    # set lr scheduler method
    if configs.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    elif configs.lr_scheduler == "on_loss":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    elif configs.lr_scheduler == "on_acc":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=False)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
    # for fp16
    if configs.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=configs.opt_level,
                                          keep_batchnorm_fp32= None if configs.opt_level == "O1" else configs.keep_batchnorm_fp32
                                          )
    if configs.resume:
            # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(configs.resume), 'Error: no checkpoint directory found!'
        configs.checkpoint = os.path.dirname(configs.resume)
        checkpoint = torch.load(configs.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(configs.log_dir, '%s_log.txt'%configs.model_name), title=configs.model_name, resume=True)
    else:
        logger = Logger(os.path.join(configs.log_dir, '%s_log.txt'%configs.model_name), title=configs.model_name)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    if configs.evaluate:
        print('\nEvaluation only')
        val_loss, val_acc = validate(val_loader, model, criterion, start_epoch)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (val_loss, val_acc))
        return

    # Train and val
    for epoch in range(start_epoch, configs.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, configs.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_acc, test_5 = validate(val_loader, model, criterion, epoch)
        # adjust lr
        if configs.lr_scheduler == "on_loss":
            scheduler.step(val_loss)
        elif configs.lr_scheduler == "on_acc":
            scheduler.step(val_acc)
        elif configs.lr_scheduler == "step":
            scheduler.step(epoch)
        elif configs.lr_scheduler == "adjust":
            adjust_learning_rate(optimizer,epoch)
        else:
            scheduler.step(epoch)
        # append logger file
        lr_current = get_lr(optimizer)
        logger.append([lr_current,train_loss, val_loss, train_acc, val_acc])
        print('train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, val_loss, train_acc, train_5, val_acc, test_5))

        # save model
        is_best = val_acc > best_acc
        is_best_loss = val_loss < best_loss
        best_acc = max(val_acc, best_acc)
        best_loss = min(val_loss,best_loss)

        save_checkpoint({
            'fold': 0,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'acc': val_acc,
            'best_acc': best_acc,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best,is_best_loss)

    logger.close()
    print('Best acc:')
    print(best_acc)
def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Training: ', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if configs.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def validate(val_loader, model, criterion, epoch):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Validating: ', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    main()
