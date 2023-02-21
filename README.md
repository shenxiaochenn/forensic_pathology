# forensic_pathology











##usage
```python
train contrast learning, you need eight NVIDIA GEFORCE RTX 3090 Graphics Cards
python -m torch.distributed.launch --nproc_per_node=8 --master_port=xxxx  train_main.py --epochs=100 --batch_size_pergpu=128 --obj_loss=True | tee xxx.log

```
