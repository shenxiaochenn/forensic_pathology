import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import wraps
import copy
from backbone import shenxiaochenNet
import torch.distributed as dist

net=shenxiaochenNet()

class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ObjectNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8,
                 norm_layer=None,
                 obj_loss=True,
                 **kwargs):
        super(ObjectNeck, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)

        self.obj_loss = obj_loss

    def forward(self, x):
        b, c, h, w = x.shape

        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x = x.flatten(2)  # (bs, c, h*w)
        if self.obj_loss:
            z = self.proj(torch.cat([x_pool, x], dim=2))  # (bs, k, 1+h*w)
            z_g, obj_attn = torch.split(z, [1, x.shape[2]], dim=2)  # (bs, nH*k, 1), (bs, nH*k, h*w)

            # do attention according to obj attention map
            obj_attn = F.normalize(obj_attn, dim=1) if self.l2_norm else obj_attn
            obj_attn /= self.scale
            obj_attn = F.softmax(obj_attn, dim=2)
            obj_attn = obj_attn.view(b, self.num_heads, -1, h * w)
            x = x.view(b, self.num_heads, -1, h * w)
            obj_val = torch.matmul(x, obj_attn.transpose(3, 2))  # (bs, nH, c//Nh, k)
            obj_val = obj_val.view(b, c, obj_attn.shape[-2])  # (bs, c, k)

            # projection
            obj_val = self.proj_obj(obj_val)  # (bs, k, k)
        else:
            z_g = self.proj(x_pool)  # (bs, k, 1)
            obj_val = None


        return z_g, obj_val  # (bs, k, 1), (bs, k, k), where the second dim is channel



class EncoderObj(nn.Module):
    def __init__(self, in_dim_neck,hid_dim, out_dim, norm_layer=None, num_mlp=2,
                 scale=1., l2_norm=True, num_heads=4,obj_loss=True):
        super(EncoderObj, self).__init__()
        self.backbone = net
        in_dim = self.backbone.out_channels
        self.project = nn.Sequential(nn.Conv2d(in_dim,in_dim_neck,kernel_size=1,padding=0,stride=1), nn.BatchNorm2d(in_dim_neck), nn.ReLU(inplace=True))
        self.neck = ObjectNeck(in_channels=in_dim_neck, hid_channels=hid_dim, out_channels=out_dim,
                               norm_layer=norm_layer, num_layers=num_mlp,
                               scale=scale, l2_norm=l2_norm, num_heads=num_heads,obj_loss=obj_loss)



    def forward(self, im):
        out = self.backbone(im)
        out = self.project(out)
        out = self.neck(out)
        return out


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class BYOL(nn.Module):
    def __init__(self, in_dim_neck=2048,dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2,
                 scale=1., l2_norm=True, num_heads=4, loss_weight_gamma=50,loss_weight_theta=0.5,obj_loss=False, **kwargs):
        super().__init__()

        self.base_m = m

        self.loss_weight_gamma = loss_weight_gamma
        self.loss_weight_theta = loss_weight_theta


        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = EncoderObj(in_dim_neck,hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads,obj_loss=obj_loss)
        self.target_net = None
        self.target_ema_updater = EMA(self.base_m)

        self.predictor = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)

        self.predictor_obj = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)

        self.obj_loss = obj_loss


    @singleton('target_net')
    def _get_target_encoder(self):
        target_net = copy.deepcopy(self.online_net)
        set_requires_grad(target_net, False)
        return target_net

    def update_par(self):
        assert self.target_net is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_net, self.online_net)

    def forward(self, x, **kwargs):
        """
        Input:
             x ( a list)
        Output:
            loss
        """
        # compute online_net features
        if self.obj_loss:
            o_proj_z_one, o_proj_obj_one = self.online_net(x[0])
            o_proj_z_two, o_proj_obj_two = self.online_net(x[1])
            b, c, n = o_proj_obj_one.shape
            obj_o_pred_one = self.predictor_obj(o_proj_obj_one).transpose(2, 1).reshape(b * n, c).contiguous()
            obj_o_pred_two = self.predictor_obj(o_proj_obj_two).transpose(2, 1).reshape(b * n, c).contiguous()
            obj_o_pred_one = torch.cat(FullGatherLayer.apply(obj_o_pred_one), dim=0)
            obj_o_pred_two = torch.cat(FullGatherLayer.apply(obj_o_pred_two), dim=0)
        else:
            o_proj_z_one, _ = self.online_net(x[0])
            o_proj_z_two, _ = self.online_net(x[1])
        online_pred_one = self.predictor(o_proj_z_one).squeeze(-1).contiguous()
        online_pred_two = self.predictor(o_proj_z_two).squeeze(-1).contiguous()
        online_pred_one = torch.cat(FullGatherLayer.apply(online_pred_one), dim=0)
        online_pred_two = torch.cat(FullGatherLayer.apply(online_pred_two), dim=0)

        # compute target_net features
        with torch.no_grad():  # no gradient to keys
            self.target_net = self._get_target_encoder()
            if self.obj_loss:
                t_proj_z_one, t_proj_obj_one = self.target_net(x[0])
                t_proj_z_two, t_proj_obj_two = self.target_net(x[1])
                b_t, c_t, n_t = t_proj_obj_one.shape
                obj_t_proj_one = t_proj_obj_one.transpose(2, 1).reshape(b_t * n_t, c_t).contiguous()
                obj_t_proj_two = t_proj_obj_two.transpose(2, 1).reshape(b_t * n_t, c_t).contiguous()
                obj_t_proj_one = concat_all_gather(obj_t_proj_one)
                obj_t_proj_two = concat_all_gather(obj_t_proj_two)
                obj_t_proj_one.detach_()
                obj_t_proj_two.detach_()
            else:
                t_proj_z_one, _ = self.target_net(x[0])
                t_proj_z_two, _ = self.target_net(x[1])
            target_proj_one = t_proj_z_one.squeeze(-1).contiguous()
            target_proj_two = t_proj_z_two.squeeze(-1).contiguous()
            target_proj_one = concat_all_gather(target_proj_one)
            target_proj_two = concat_all_gather(target_proj_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        # instance loss
        loss_ins_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_ins_two = loss_fn(online_pred_two, target_proj_one.detach())
        # object loss
        if self.obj_loss:
            loss_obj_one = loss_fn(obj_o_pred_one, obj_t_proj_two.detach())
            loss_obj_two = loss_fn(obj_o_pred_two, obj_t_proj_one.detach())

        # variance loss
        online_pred_one = online_pred_one - online_pred_one.mean(dim=0)
        std_online_pred_one = torch.sqrt(online_pred_one.var(dim=0) + 0.0001)
        std_loss_one = torch.mean(F.relu(1 - std_online_pred_one))
        online_pred_two = online_pred_two - online_pred_two.mean(dim=0)
        std_online_pred_two = torch.sqrt(online_pred_two.var(dim=0) + 0.0001)
        std_loss_two = torch.mean(F.relu(1 - std_online_pred_two))

        # cov loss
        b_1, f_1 = online_pred_one.shape
        cov_pred_one = (online_pred_one.T @ online_pred_one) / (b_1 - 1)
        cov_loss_one = off_diagonal(cov_pred_one).pow_(2).clamp_(1e-5,10.0).sum().div(f_1)
        b_2, f_2 = online_pred_two.shape
        cov_pred_two = (online_pred_two.T @ online_pred_two) / (b_2 - 1)
        cov_loss_two = off_diagonal(cov_pred_two).pow_(2).clamp_(1e-5,10.0).sum().div(f_2)

        # all loss
        loss_ins = (loss_ins_one + loss_ins_two).mean()
        loss_obj = (loss_obj_one + loss_obj_two).mean()
        loss_std = (std_loss_one + std_loss_two).mean() * self.loss_weight_gamma
        loss_cov = (cov_loss_one + cov_loss_two).mean() * self.loss_weight_theta
        loss = (loss_ins + loss_obj+loss_cov+loss_std) / 4

        return loss, loss_ins, loss_cov




def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    other = torch.cat(tensors_gather[:rank] + tensors_gather[rank+1:], dim=0)
    return other



@torch.no_grad()
def concat_all_gather(tensor, replace=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    if replace:
        tensors_gather[rank] = tensor
    other = torch.cat(tensors_gather, dim=0)
    return other

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


if __name__ == '__main__':
    from lars import LARS
    #print("hahahaha")
    #print(range(2))
    mm = BYOL()
    mm.eval()
    mm._get_target_encoder()
    mm.target_net=mm._get_target_encoder()
    def exclude_bias_and_norm(p):
        return p.ndim == 1
    checkpoint = torch.load("FPath-self-obj-100.pth")
    mm.load_state_dict(checkpoint['model'])
