import sys
sys.path.append("/home/hpcstack/shenxiaochen/pathology")
#sys.path.append("/home/shenxiaochen/pathology/new")
import os
import time
import torch
from torch import nn
from torchvision import transforms, datasets
from myTransforms import HEDJitter,RandomAffineCV2,RandomGaussBlur
from utils import TwoCropTransform, AverageMeter
from lars import LARS
import math
import numpy as np
from lossplot import LossHistory
import argparse
from model import BYOL  #这里要改注意一下
from PIL import ImageFilter
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--local_rank', type=int, help='local rank for dist')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--data_folder', type=str, default='/home/hpcstack/data/shen_chen_path_mouse/data/train_data', help='path to custom dataset')
parser.add_argument('--pretrain_encoder_one', type=str, default='resnet50.pth', help='https://github.com/facebookresearch/barlowtwins')
parser.add_argument('--pretrain_encoder_two', type=str, default='moby_swin_t_300ep_pretrained.pth', help='https://github.com/SwinTransformer/Transformer-SSL')
parser.add_argument('--batch_size_pergpu', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
parser.add_argument('--size', type=int, default=224, help='parameter for RandomCrop')
parser.add_argument("--base_lr", type=float, default=0.02,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--num_heads', type=int, default=4, help='number heads of object loss')
parser.add_argument('--gamma', type=int, default=5, help='hyperparameter of std loss')
parser.add_argument('--theta', type=int, default=0.0051, help='hyperparameter of cov loss')
parser.add_argument('--obj_loss', type=bool, default=True, help='learn where to learn')

args = parser.parse_args()

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

train_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomChoice([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), HEDJitter(theta=0.05)])
    ], p=0.3),
    transforms.RandomApply([RandomAffineCV2(alpha=0.1)], p=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([RandomGaussBlur(radius=[0.5, 1.5])], p=0.3),
    GaussianBlur(p=0.3),
    transforms.RandomResizedCrop(size=args.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]),   #这里重新计算了
])


def adjust_learning_rate(args, optimizer, loader, step,world_size):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size_pergpu*world_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train(train_loader, model, optimizer, epoch,local_rank,scaler,world_size):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    loss3 = AverageMeter()
    loss4 = AverageMeter()

    end = time.time()
    for step, (images, labels) in enumerate(train_loader, start=epoch * len(train_loader)):
        lr = adjust_learning_rate(args,optimizer,train_loader,step,world_size)
        data_time.update(time.time() - end)


        if torch.cuda.is_available() and local_rank is not None:
            images = [images[0].cuda(local_rank, non_blocking=True), images[1].cuda(local_rank, non_blocking=True)]

        bsz = labels.shape[0]
        optimizer.zero_grad()
        # compute loss
        with torch.cuda.amp.autocast():
            loss,loss_ins,loss_obj,loss_std,loss_cov = model(images)

        # update metric
        losses.update(loss.item(), bsz)
        loss1.update(loss_ins.item(), bsz)
        loss2.update(loss_obj.item(), bsz)
        loss3.update(loss_std.item(), bsz)
        loss4.update(loss_cov.item(), bsz)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (step + 1) % args.print_freq == 0 and local_rank == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t''loss_ins {loss_ins.val:.3f} ({loss_ins.avg:.3f})\t''loss_obj {loss_obj.val:.3f} ({loss_obj.avg:.3f})\t'
                  'loss_std {loss_std.val:.3f} ({loss_std.avg:.3f})\t''loss_cov {loss_cov.val:.3f} ({loss_cov.avg:.3f})\t'
                  'LR {lr: .3f}'.format(
                epoch, ((step + 1) - (epoch * len(train_loader))), len(train_loader), loss=losses,loss_ins=loss1,loss_obj=loss2,loss_std=loss3,loss_cov=loss4,
                lr=lr))
            sys.stdout.flush()

    return losses.avg,loss1.avg,loss2.avg,loss3.avg,loss4.avg


def exclude_bias_and_norm(p):
    return p.ndim == 1

loss_all = [10]

def main():
    from torch.nn.parallel import DistributedDataParallel
    torch.backends.cudnn.benchmark = True

    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        loss_history = LossHistory("./logs_self/",val_loss_flag=True,test_loss_flag=True,test2_loss_flag=True)


    model = BYOL(num_heads=args.num_heads,loss_weight_gamma=args.gamma,loss_weight_theta=args.theta,obj_loss=args.obj_loss,)
    stat_dic1 = torch.load(args.pretrain_encoder_one,map_location="cpu")
    stat_dic2 = torch.load(args.pretrain_encoder_two,map_location="cpu")
    # model_dic = {k[15:]: v for k, v in stat_dic1["model"].items() if k[7:] in model.online_net.backbone.state_dict().keys()}
    model_dic2 = {k[8:]: v for k, v in stat_dic2["model"].items() if k[8:] in model.online_net.backbone.encoder2.state_dict().keys()}
    model.online_net.backbone.encoder.load_state_dict(stat_dic1, strict=True)
    model.online_net.backbone.encoder2.load_state_dict(model_dic2, strict=True)
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    for pp in [model.module.online_net.backbone.encoder.conv1, model.module.online_net.backbone.encoder.bn1, model.module.online_net.backbone.encoder.layer1,
               model.module.online_net.backbone.encoder.layer2, model.module.online_net.backbone.encoder2.patch_embed, model.module.online_net.backbone.encoder2.layers[0],
               model.module.online_net.backbone.encoder2.layers[1]]:
        for param in pp.parameters():
            param.requires_grad = False

    pg = [p for p in model.module.parameters() if p.requires_grad]


    #    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.learning_rate)
    optimizer = LARS(
        pg,
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )
    #if local_rank==0:
        #print("pg",len(pg))
        #print("params",optimizer.state_dict()["param_groups"])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_folder),
        transform=TwoCropTransform(train_transform))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_pergpu, num_workers=args.num_workers,
                                              pin_memory=True, sampler=train_sampler,drop_last=True)


    checkpoint_path = 'checkpoints/FPath-self-instance-objcovstd-{}.pth'.format(args.epochs) ################要改
    if local_rank == 0:  #打印定义的参数
        print('checkpoint_path:', checkpoint_path)
        for k, v in sorted(vars(args).items()):
            print(k, '=', v)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.module.target_net = model.module._get_target_encoder()
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    model.train()

    for param in model.module.online_net.backbone.parameters():
        param.requires_grad = False
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss,loss_ins,loss_obj,loss_std,loss_cov = train(trainloader, model, optimizer, epoch,local_rank,scaler,world_size)
        if local_rank == 0:
            loss_history.append_loss(loss=loss,val_loss=loss_ins,test_loss=loss_obj,test2_loss=loss_cov)
        model.module.update_par()
        if epoch == 5:
            for param in model.module.online_net.backbone.parameters():
                param.requires_grad = True
            for pp in [model.module.online_net.backbone.encoder.conv1, model.module.online_net.backbone.encoder.bn1,
                       model.module.online_net.backbone.encoder.layer1,
                       model.module.online_net.backbone.encoder.layer2,
                       model.module.online_net.backbone.encoder2.patch_embed,
                       model.module.online_net.backbone.encoder2.layers[0],
                       model.module.online_net.backbone.encoder2.layers[1]]:
                for param in pp.parameters():
                    param.requires_grad = False

        # check
        model_state_dict = model.module.state_dict()
        for k in model_state_dict:
            if k.startswith("online_net.backbone.encoder.conv1.weight"):
                assert torch.equal(stat_dic1[k.replace('online_net.backbone.encoder.', '')].cpu(),
                                   model_state_dict[k].cpu()), "wrong1"


            if k.startswith("online_net.backbone.encoder2.layers.1.blocks.1.attn.qkv.weight"):
                assert torch.equal(stat_dic2["model"][k.replace('online_net.backbone.encoder2', 'encoder')].cpu(),
                                   model_state_dict[k].cpu()),  "wrong2"

            # if k.startswith("online_net.backbone.encoder.layer3.0.conv3.weight"):
            #     assert torch.equal(stat_dic1[k.replace('online_net.backbone.encoder.', '')].cpu(),
            #                        model_state_dict[k].cpu()), "wrong3"




            # if epoch % 2 == 0:
            #     for param in model.module.online_net.backbone.encoder2.parameters():
            #         param.requires_grad = False
            #     for param in model.module.online_net.backbone.encoder.parameters():
            #         param.requires_grad = True
            # else:
            #     for param in model.module.online_net.backbone.encoder2.parameters():
            #         param.requires_grad = True
            #     for param in model.module.online_net.backbone.encoder.parameters():
            #         param.requires_grad = False


        if epoch > 5 and loss < min(loss_all) and local_rank == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)
            loss_all.append(loss)





if __name__ == "__main__":
    main()
