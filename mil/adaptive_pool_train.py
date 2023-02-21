import os
import sys
sys.path.append("/home/hpcstack/shenxiaochen/pathology")
import torch
import torch.nn as nn
from model import BYOL
from myTransforms import HEDJitter,RandomAffineCV2
from torchvision import transforms, datasets
from PIL import ImageFilter
import numpy as np
from timm.models.layers import trunc_normal_
from einops.layers.torch import Rearrange
from swin_relative_trans import Block

import random

from utils import AverageMeter, accuracy
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC,MulticlassRecall,MulticlassPrecision,MulticlassAccuracy,MulticlassF1Score,MulticlassMatthewsCorrCoef
import argparse
parser = argparse.ArgumentParser('argument for multi instance learning')
parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
parser.add_argument('--print_freq', type=int, default=10,help='print frequency')

parser.add_argument('--checkpoint', type=str, default="/home/hpcstack/shenxiaochen/pathology/checkpoints/FPath-linear-evaluate-objstdcov-50.pth", help='evaluate checkpoint')

parser.add_argument('--data_folder', type=str, default='/home/hpcstack/data/', help='path to custom dataset')
parser.add_argument('--size', type=int, default=(896,1344), help='parameter for RandomResizedCrop')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=0.005,help='learning rate')
parser.add_argument('--schedule', default=[15, 30,40,80], type=int,help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#数据
# 数据位置
traindir = args.data_folder + "train_human"
valdir = args.data_folder + "dev_human"
testdir = args.data_folder + "test_human"
def seed_torch(seed=1412):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# 数据增强
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
    # transforms.RandomApply([
    #     transforms.RandomChoice([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), HEDJitter(theta=0.05)])
    # ], p=0.7),
    # transforms.RandomApply([RandomAffineCV2(alpha=0.1)], p=0.7),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(p=0.3),
    GaussianBlur(p=0.3),
    transforms.RandomResizedCrop(size=args.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]),
])
valid_transform = transforms.Compose([
        # transforms.RandomApply([
        #     transforms.RandomChoice([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), HEDJitter(theta=0.05)])
        # ], p=0.7),
        # transforms.RandomApply([RandomAffineCV2(alpha=0.1)], p=0.7),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.3),
        # GaussianBlur(p=0.7),
        transforms.Resize(896),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]),
     ])
test_transform = transforms.Compose([
        # transforms.RandomApply([
        #     transforms.RandomChoice([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), HEDJitter(theta=0.05)])
        # ], p=0.7),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.3),
        # GaussianBlur(p=0.7),
        transforms.Resize(896),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]),
     ])

train_dataset = datasets.ImageFolder(
    root=traindir,
    transform=train_transform)

valid_dataset = datasets.ImageFolder(
    root=valdir,
    transform=test_transform)
test_dataset = datasets.ImageFolder(
    root=testdir,
    transform=test_transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,
        num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,
        num_workers=args.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,
        num_workers=args.num_workers, pin_memory=True)


# 模型
model = BYOL()
backbone = model.online_net.backbone
pool_layer = nn.AdaptiveAvgPool2d(1)
flatten_ = torch.nn.Flatten(1)
head = nn.Linear(backbone.out_channels, args.num_classes)
model = nn.Sequential(backbone, pool_layer, flatten_, head)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['model'])
backbone = model[0].to(device)

# 相对位置编码+adaptive池化
class adaptivepool(torch.nn.Module):
    def __init__(self,  dim=128):
        super(adaptivepool, self).__init__()
        self.proj_att = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # （b,n,c）
        x_ = x
        x = self.proj_att(x)
        x = x.transpose(1, 2)
        x = x.softmax(dim=-1)
        x_ = self.proj_v(x_)
        x_ = x_.transpose(1, 2)
        x = x * x_
        x = self.avgpool(x)

        return x
class Lineartransformer(nn.Module):
    """ transformer"""

    def __init__(self, dimm=2816, feat_dim=128, heads=16,num_classes=args.num_classes):
        super(Lineartransformer, self).__init__()
        self.pro = nn.Linear(dimm, feat_dim)
        self.block = nn.Sequential(Block(dim=feat_dim,num_heads=heads))
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp_head = nn.Linear(feat_dim, num_classes)

        self.adaptivepool = nn.Sequential(adaptivepool())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):  # (b,24,2816)
        x = self.pro(x)
        x = self.norm1(x)
        x = self.block(x)
        x = self.norm2(x)
        x = self.adaptivepool(x)
        x = torch.flatten(x, 1)
        x = self.mlp_head(x)

        return x


classifier = Lineartransformer()

pretrained_dict = torch.load('./checkpoints/MILL-adaptive-human-50.pth',map_location="cpu")["model"]
# model_dict = classifier.state_dict()
# pretrained_dict2 = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v)}
# model_dict.update(pretrained_dict2)
# classifier.load_state_dict(model_dict)
#classifier.load_state_dict(torch.load('/home/hpcstack/shenxiaochen/pathology/checkpoints/MIL-relative-{}.pth'.format(args.epochs),map_location="cpu")["model"])
classifier.load_state_dict(pretrained_dict)
classifier = classifier.to(device)
##  损失函数。。优化器。。。学习率策略
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = torch.optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=0.9,
                          weight_decay=0)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
##############################################
patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=224, p2=224), #图片切分重排
        ).to(device)
re_patch = nn.Sequential(pool_layer,flatten_,Rearrange('(b h w) c -> b (h w) c', h=4, w=6)).to(device)


def main():

    metric_collection = MetricCollection({
        'acc': MulticlassAccuracy(num_classes=args.num_classes),
        'precision': MulticlassPrecision(num_classes=args.num_classes),
        'recall': MulticlassRecall(num_classes=args.num_classes),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=args.num_classes),
        "f1-score": MulticlassF1Score(num_classes=args.num_classes),
        "AUC": MulticlassAUROC(num_classes=args.num_classes)

    }).to(device)
    checkpoint_path = './checkpoints/MILL-adaptive-human2-{}.pth'.format(args.epochs)  # here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('checkpoint_path:', checkpoint_path)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    acc_val_all = [50]
    acc_test_all = [50]
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        losses = AverageMeter()
        for step, (images, target) in enumerate(
                train_loader, start=epoch * len(train_loader)):
            images = images.to(device, non_blocking=True)
            labels = target.to(device, non_blocking=True)
            bsz = labels.shape[0]
            # compute loss
            backbone.eval()
            classifier.train()
            with torch.no_grad():
                images = patch(images)
                features = backbone(images)
                features = re_patch(features)  # features (B N C)

            output = classifier(features.detach())
            loss = criterion(output, labels)
            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'lr {lr: .3f}\t'.format(
                    epoch, ((step + 1) - (epoch * len(train_loader))), len(train_loader), loss=losses, lr=lr,
                    ))
                sys.stdout.flush()
        backbone.eval()
        classifier.eval()
        top1_val = AverageMeter()
        with torch.no_grad():
            for images, target in val_loader:
                images = images.to(device, non_blocking=True)
                labels = target.to(device, non_blocking=True)
                images = patch(images)
                features = backbone(images)
                features = re_patch(features)  # features (B N C)
                output = classifier(features.detach())
                acc1, = accuracy(
                    output, labels, topk=(1,)
                )
                top1_val.update(acc1[0].item(), labels.size(0))
        print(f"Validation acc for epoch {epoch + 1}: {top1_val.avg}",flush=True)


        top1_test = AverageMeter()
        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device, non_blocking=True)
                labels = target.to(device, non_blocking=True)
                images = patch(images)
                features = backbone(images)
                features = re_patch(features)  # features (B N C)
                output = classifier(features.detach())
                acc1, = accuracy(
                    output, labels, topk=(1,)
                )
                top1_test.update(acc1[0].item(), labels.size(0))
        print(f"test acc for epoch {epoch + 1}: {top1_test.avg}", flush=True)



        if top1_val.avg > max(acc_val_all):
            torch.save(
                {
                    'model': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)
            acc_val_all.append(top1_val.avg)
            acc_test_all.append(top1_test.avg)
#########################  结果的打印！！！！#############################
    print("best acc_val:",max(acc_val_all),"------best acc_test:",max(acc_test_all))
    print("------------------------------------------------------------------------")
    classifier.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    seed_torch(1412)
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device, non_blocking=True)
            labels = target.to(device, non_blocking=True)
            images = patch(images)
            features = backbone(images)
            features = re_patch(features)  # features (B N C)
            output = classifier(features.detach())
            metric_collection.update(output.softmax(dim=-1), labels.flatten())
    val_matrix = metric_collection.compute()
    print("valid matrix:", val_matrix)
    metric_collection.reset()
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device, non_blocking=True)
            labels = target.to(device, non_blocking=True)
            images = patch(images)
            features = backbone(images)
            features = re_patch(features)  # features (B N C)
            output = classifier(features.detach())
            metric_collection.update(output.softmax(dim=-1), labels.flatten())
    test_matrix = metric_collection.compute()
    print("test matrix: ", test_matrix)
    metric_collection.reset()




if __name__ == '__main__':
    main()



