import os
import sys
sys.path.append("/home/hpcstack/shenxiaochen/pathology")
#sys.path.append("/home/shenxiaochen/pathology/new")
import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import pandas as pd
from model import BYOL
from utils import AverageMeter, accuracy
import numpy as np
from PIL import ImageFilter
from torch.nn.parallel import DistributedDataParallel
parser = argparse.ArgumentParser('argument for evaluating')
parser.add_argument('--local_rank', type=int, help='local rank for dist')
parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
parser.add_argument('--print_freq', type=int, default=10,help='print frequency')
parser.add_argument('--data_folder', type=str, default='/home/hpcstack/data/patch_human/', help='path to custom dataset')
parser.add_argument('--batch_size_pergpu', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
parser.add_argument('--train_percent', type=float, default=1.0, help='train dataset percent')
parser.add_argument('--backbone_checkpoint', type=str, default="FPath-self-instance-objcovstd-100.pth", help='contrast learning checkpoint')  #here!!!!!!!!!!
parser.add_argument('--save_checkpoint', type=str, default="ins", help='name of saved checkpoint')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"), help="finetune or freeze resnet weights")
parser.add_argument("--lr_backbone", default=0.0, type=float, help="backbone base learning rate")
parser.add_argument("--lr_head", default=0.3, type=float, help="classifier base learning rate")
#parser.add_argument('--schedule', default=[10, 30, 40], type=int,help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
args = parser.parse_args()

def main():

    # 数据位置
    traindir = args.data_folder + "train_patch_human"
    valdir = args.data_folder + "dev_patch_human"
    testdir = args.data_folder + "test_patch_human"

    acc_val_all = [50]
    acc_test_all = [50]
    torch.backends.cudnn.benchmark = True
    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    normalize = transforms.Normalize(
        mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]
    )

    class GaussianBlur(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, img):
            if np.random.rand() < self.p:
                sigma = np.random.rand() * 1.9 + 0.1
                return img.filter(ImageFilter.GaussianBlur(sigma))
            else:
                return img

    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [GaussianBlur(p=0.7),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )


    # 模型
    model = BYOL()
    model.target_net = model._get_target_encoder()


    checkpoint = torch.load("checkpoints/" + args.backbone_checkpoint,map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    backbone = model.online_net.backbone

    pool_layer = nn.AdaptiveAvgPool2d(1)
    flatten_ = torch.nn.Flatten(1)
    head = nn.Linear(backbone.out_channels, args.num_classes)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, pool_layer, flatten_, head)
    model = model.cuda()

    if args.weights == "freeze":
        backbone.requires_grad_(False)

        head.requires_grad_(True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias


    optimizer = optim.SGD(parameters, args.lr_head, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # def adjust_learning_rate(optimizer, epoch, args):
    #     """Decay the learning rate based on schedule"""
    #     lr = args.lr_head
    #     for milestone in args.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     return lr

    checkpoint_path = 'checkpoints/FPath-linear-evaluate-human22-{}-{}.pth'.format(args.save_checkpoint, args.epochs) #here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if args.local_rank == 0:  # 打印定义的参数
        print('checkpoint_path:', checkpoint_path)
        for k, v in sorted(vars(args).items()):
            print(k, '=', v)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    kwargs = dict(
        batch_size=args.batch_size_pergpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        losses = AverageMeter()
        for step, (images, target) in enumerate(
                train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(args.local_rank, non_blocking=True))
            loss = criterion(output, target.cuda(args.local_rank, non_blocking=True))
            bsz = target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update metric
            torch.distributed.reduce(loss.div_(world_size), 0)
            losses.update(loss.item(), bsz)
            if (step + 1) % args.print_freq == 0 and args.local_rank == 0:

                lr_head = lr
                lr_backbone = 0
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'lr_backbone {lr_backbone: .3f}\t''lr_head {lr_head: .3f}\t'.format(
                    epoch, ((step + 1) - (epoch * len(train_loader))), len(train_loader), loss=losses, lr_backbone=lr_backbone,
                    lr_head=lr_head))
                sys.stdout.flush()
        # evaluate
        model.eval()
        if args.local_rank == 0:
            top1_val = AverageMeter()
            top1_test = AverageMeter()
            with torch.no_grad():
                for images, target in val_loader:

                    output = model(images.cuda(args.local_rank, non_blocking=True))
                    acc1, = accuracy(
                        output, target.cuda(args.local_rank, non_blocking=True), topk=(1,)
                    )
                    top1_val.update(acc1[0].item(), images.size(0))

                for images, target in test_loader:

                    output = model(images.cuda(args.local_rank, non_blocking=True))
                    acc1, = accuracy(
                        output, target.cuda(args.local_rank, non_blocking=True), topk=(1,)
                    )
                    top1_test.update(acc1[0].item(), images.size(0))
            print('Epoch: [{0}]\t'
                  'val_ACC {top1_val.val:.3f} ({top1_val.avg:.3f})\t'
                  'test_ACC {top1_test.val:.3f} ({top1_test.avg:.3f})\t'.format(epoch,top1_val=top1_val,top1_test=top1_test))
            if epoch > 2 and top1_val.avg > max(acc_val_all):
                torch.save(
                    {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1
                    }, checkpoint_path)
                acc_val_all.append(top1_val.avg)
                acc_test_all.append(top1_test.avg)

        scheduler.step()

        if args.weights == 'freeze':
            reference_state_dict = torch.load("checkpoints/" + args.backbone_checkpoint,map_location='cpu')
            model_state_dict = model.module.state_dict()
            for k in reference_state_dict["model"]:
                if k.startswith("online_net.backbone"):
                    assert torch.equal(model_state_dict[k.replace('online_net.backbone', '0')].cpu(),
                                       reference_state_dict["model"][k]), k








if __name__ == "__main__":
    def exclude_bias_and_norm(p):  #这个有用。。要不你load不了checkpoint
        return p.ndim == 1
    main()
