import os
import sys
import torch
sys.path.append("/home/hpcstack/shenxiaochen/pathology")
import argparse
from torchvision import transforms, datasets

from model import BYOL
from backbone import resnet50
from backbone import swin_tiny_patch4_window7_224
from myTransforms import HEDJitter,RandomAffineCV2
from PIL import ImageFilter
import numpy as np
import random
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC,MulticlassRecall,MulticlassPrecision,MulticlassAccuracy,MulticlassF1Score,MulticlassMatthewsCorrCoef



parser = argparse.ArgumentParser('argument for evaluating')

parser.add_argument('--data_folder', type=str, default='/home/hpcstack/data/patch_human/', help='path to custom dataset')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
parser.add_argument('--checkpoint', type=str, default="checkpoints/FPath-linear-evaluate-human22-ins22-50.pth", help='evaluate checkpoint')

args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seed_torch(seed=1412):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    seed_torch(1412)

    # 数据位置

    valdir = args.data_folder + "dev_patch_human"
    testdir = args.data_folder + "test_patch_human"



    class GaussianBlur(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, img):
            if np.random.rand() < self.p:
                sigma = np.random.rand() * 1.9 + 0.1
                return img.filter(ImageFilter.GaussianBlur(sigma))
            else:
                return img

    normalize = transforms.Normalize(
        mean=[0.848, 0.524, 0.754], std=[0.117, 0.180, 0.127]
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [   GaussianBlur(0.2),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose(
            [   GaussianBlur(0.2),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    kwargs = dict(
        batch_size=96,
        num_workers=6,
        pin_memory=True,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    # 模型
    model = BYOL()
    backbone = model.online_net.backbone
    pool_layer = nn.AdaptiveAvgPool2d(1)
    flatten_ = torch.nn.Flatten(1)
    head = nn.Linear(backbone.out_channels, args.num_classes)
    # model = nn.Sequential(backbone, pool_layer, flatten_, head)
    # checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # model = model.to(device)
    # balow twins or swin
    #backbone = resnet50()
    # backbone = swin_tiny_patch4_window7_224()
    # pool_layer = nn.AdaptiveAvgPool2d(1)
    # flatten_ = torch.nn.Flatten(1)
    # head = nn.Linear(768, args.num_classes)
    model = nn.Sequential(backbone, pool_layer, flatten_, head)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)


    metric_collection = MetricCollection({
        'acc': MulticlassAccuracy(num_classes=args.num_classes),
        'precision': MulticlassPrecision(num_classes=args.num_classes),
        'recall': MulticlassRecall(num_classes=args.num_classes),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=args.num_classes),
        "f1-score": MulticlassF1Score(num_classes=args.num_classes),
        "AUC": MulticlassAUROC(num_classes=args.num_classes)

    }).to(device)

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            preds = model(images.to(device, non_blocking=True))
            preds = preds.softmax(dim=-1)
            target = target.to(device, non_blocking=True).flatten()
            metric_collection.update(preds,target)
        val_metrics = metric_collection.compute()
        print(f"Metrics on dev all data: {val_metrics}")
        metric_collection.reset()
        for i, (images, target) in enumerate(test_loader):
            preds = model(images.to(device, non_blocking=True))
            preds = preds.softmax(dim=-1)
            target = target.to(device, non_blocking=True).flatten()
            metric_collection.update(preds, target)
        test_metrics = metric_collection.compute()
        print(f"Metrics on test all data: {test_metrics}")
        metric_collection.reset()









if __name__ == "__main__":
    main()



