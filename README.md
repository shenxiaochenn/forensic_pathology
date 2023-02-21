# forensic_pathology











## usage
```python
train contrast learning, you need eight NVIDIA GEFORCE RTX 3090 Graphics Cards
python -m torch.distributed.launch --nproc_per_node=8 --master_port=xxxx  train_main.py --epochs=100 --batch_size_pergpu=128 --obj_loss=True | tee xxx.log

```
```python
linear evaluation (percent or all! if all data train_percent=1)
CUDA_VISIBLE_DEVICES=xxxx python -m torch.distributed.launch --nproc_per_node=8 --master_port=xxxx   linear_percent.py --train_percent=xxx  --save_checkpoint=xxx --weights=freeze  | tee xxx.log
```
