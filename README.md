# forensic_pathology
****
![image](https://github.com/shenxiaochenn/forensic_pathology/blob/master/fig/framework.png)
****







## require
```
pytorch (cuda=11.7)
torchvision
torchmetrics 
timm
einops
cython
scikit-image
opencv-python 
scikit-learn 
transformers
tqdm
```


## usage

### train contrast learning
```python
train contrast learning, you need eight NVIDIA GEFORCE RTX 3090 Graphics Cards
python -m torch.distributed.launch --nproc_per_node=8 --master_port=xxxx  train_main.py --epochs=100 --batch_size_pergpu=128 --obj_loss=True | tee xxx.log

```
### evaluate (linear)
```python
linear evaluation (percent or all! if all data train_percent=1)
CUDA_VISIBLE_DEVICES=xxxx python -m torch.distributed.launch --nproc_per_node=8 --master_port=xxxx   linear_percent.py --train_percent=xxx  --save_checkpoint=xxx --weights=freeze  | tee xxx.log
```

### train multiple instance learning
```python
when train multiple instance learning, you must have a backbone checkpoint, and also a small batch_size is required
python adaptive_pool_train.py --epochs=50 --checkpoint=xxx  --size=xxxx --batch_size=xxx
```
### visulasition
****
![iamge1](https://github.com/shenxiaochenn/forensic_pathology/blob/master/fig/20X_kidne80_0013_map.png 400*300)
****
## reference
* [vicreg](https://github.com/facebookresearch/vicreg)
* [SupContrast](https://github.com/HobbitLong/SupContrast)
* [LEWEL](https://github.com/LayneH/LEWEL)
* [barlowtwins](https://github.com/facebookresearch/barlowtwins)
## Citation

