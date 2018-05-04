import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np


def _norm(x, dim, p=2):
    """Computes the norm over all dimensions except dim"""
    if p == -1:
        func = lambda x, dim: x.max(dim=dim)[0] - x.min(dim=dim)[0]
    elif p == float('inf'):
        func = lambda x, dim: x.max(dim=dim)[0]
    else:
        func = lambda x, dim: torch.norm(x, dim=dim, p=p)
    if dim is None:
        return x.norm(p=p)
    elif dim == 0:
        output_size = (x.size(0),) + (1,) * (x.dim() - 1)
        return func(x.contiguous().view(x.size(0), -1), 1).view(*output_size)
    elif dim == x.dim() - 1:
        output_size = (1,) * (x.dim() - 1) + (x.size(-1),)
        return func(x.contiguous().view(-1, x.size(-1)), 0).view(*output_size)
    else:
        return _norm(x.transpose(0, dim), 0).transpose(0, dim)


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


def _std(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.std()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).std(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).std(dim=0).view(*output_size)
    else:
        return _std(p.transpose(0, dim), 0).transpose(0, dim)


class MyBN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        p=1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            # Var = ((t.transpose(1,0)-mean)**2).mean(0)
            Var = z.view(z.size(0),-1).var(-1, unbiased = False)
            scale = (Var+0.0000001).rsqrt()
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
            # scale = 1 / (x_flat.max(-1)[0].max(0)[0] - x_flat.min(-1)[0].min(0)[0])
        # out = x - mean.view(1, mean.size(0), 1, 1)
        # denominator = (var + 0.001).sqrt()

        out = (x - mean.view(1, mean.size(0), 1, 1))* scale.view(1, scale.size(0), 1, 1)

        # out = (x - mean.view(1, mean.size(0), 1, 1)) / \ (scale.view(1, scale.size(0), 1, 1) + 0.001)
        # out = (x - mean.view(1, mean.size(0), 1, 1)/(denominator.view(1, mean.size(0), 1, 1)))

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class TopKnoGhost(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.mean = Parameter(torch.Tensor(num_features))
        self.scale = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        # p=5
        K = 100
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)
            

            # A_total = torch.cat((OldA, A), 0)

            # scale = 1 / (A.max(0)[0] + 0.0000001)

            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / ((2 * np.log(A.size(0))) ** 0.5)

            MeanTOPK = (torch.topk(A, K, dim=0)[0].mean(0))*const

            # print(self.meanTOPK)

            # self.meanTOPK = MeanTOPK.data
            scale = 1 / (MeanTOPK + 0.0000001)

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)


        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.scale is not None:
            out = out * self.scale.view(1, self.scale.size(0), 1, 1)

        if self.mean is not None:
            out = out + self.mean.view(1, self.mean.size(0), 1, 1)
        return out


class Top1(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        self.register_buffer('meanTOPK', torch.zeros(num_features))
        self.noise = noise
        self.mean = Parameter(torch.Tensor(num_features))
        self.scale = Parameter(torch.Tensor(num_features))

        # if bias:
        #     self.mean = Parameter(torch.Tensor(num_features))
        #     self.scale = Parameter(torch.Tensor(num_features))
        # else:
        #     self.register_parameter('mean', None)
        #     self.register_parameter('scale', None)

    def forward(self, x):
        # p=5
        K=1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            A = torch.abs(t.transpose(1, 0) - mean)
            beta = 0.4
            
            # A_total = torch.cat((OldA, A), 0)

            # scale = 1 / (A.max(0)[0] + 0.0000001)
            MeanTOPK = torch.topk(A,K,dim=0)[0].mean(0)
            meanTOPK = beta*torch.autograd.variable.Variable(self.meanTOPK) + (1-beta)*MeanTOPK
            
            # print(self.meanTOPK)
            self.meanTOPK.copy_(meanTOPK.data)
            # self.meanTOPK = MeanTOPK.data
            scale = 1/(meanTOPK + 0.0000001)


            # final_scale = self.old_scale.mul_(0.5).add_(scale.data * (0.5))
            # final_scale = torch.autograd.variable.Variable(final_scale)
            # print(final_scale)

            # print(final_scale)

            # self.old_scale.copy_(scale1.data)
            # # scale = self.old_scale.mul_(0.5).add_(scale1.data * (0.5))
            # # scale = 0.5 * (self.old_scale + scale1.data)
            # scale  = scale1.data

            # self.old_scale.copy_(scale1.data)

            # scale = torch.autograd.variable.Variable(scale)

            # scale1 = scale1/scale1.mean()



            #######hybrid
        #
        # p = 1
        # if self.training:
        #     mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
        #     y = x.transpose(0, 1)
        #     z = y.contiguous()
        #     t = z.view(z.size(0), -1)
        #     Var = (torch.abs((t.transpose(1, 0) - mean)) ** p).mean(0)
        #     # Var = z.view(z.size(0),-1).var(-1, unbiased = False)
        #     # scale = (Var + 0.0000001).rsqrt()
        #
        #     scale2 =  (Var + 0.0000001) ** (-1 / p)
        #     scale2 = scale2/scale2.mean()
        #
        #     scale = 0.5*(scale1 + scale2)

            #######hybrid



            #older run
            # B = 1 / (A + 0.0000001)
            # scale = B.min(0)[0]
            # #




            # MAX =  torch.FloatTensor(torch.max(A,0))

            # scale = 1/ (MAX + 0.0000001)
            # Var = z.view(z.size(0),-1).var(-1, unbiased = False)
            # scale = (Var + 0.0000001).rsqrt()
            # scale = (Var + 0.0000001)**(-1/p)


            # scale = (Var+0.0000001).rsqrt()
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
            # final_scale = torch.autograd.Variable(self.running_var)
            # scale = 1 / (x_flat.max(-1)[0].max(0)[0] - x_flat.min(-1)[0].min(0)[0])
        # out = x - mean.view(1, mean.size(0), 1, 1)
        # denominator = (var + 0.001).sqrt()

        out = (x - mean.view(1, mean.size(0), 1, 1))* scale.view(1, scale.size(0), 1, 1)
        # out = (x - mean.view(1, mean.size(0), 1, 1)) * final_scale.view(1, scale.size(0), 1, 1)



        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.scale is not None:
            out = out * self.scale.view(1, self.scale.size(0), 1, 1)

        if self.mean is not None:
            out = out + self.mean.view(1, self.mean.size(0), 1, 1)
        return out








class MeanBN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.mean = Parameter(torch.Tensor(num_features))
        self.scale = Parameter(torch.Tensor(num_features))
        # if bias:
        #     self.mean = Parameter(torch.Tensor(num_features))
        #     self.scale = Parameter(torch.Tensor(num_features))
        # else:
        #     self.register_parameter('mean', None)
        #     self.register_parameter('scale', None)

    def forward(self, x):
        p=2
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            Var = (torch.abs((t.transpose(1,0)-mean))**p).mean(0)
            # Var = z.view(z.size(0),-1).var(-1, unbiased = False)
            # scale = (Var + 0.0000001).rsqrt()
            scale = (Var + 0.0000001)**(-1/p)
            # scale = ((Var * (np.pi / 2) ** 0.5) + 0.0000001) ** (-1 / p)
            # scale = (Var + 0.0000001) ** (-1 / p)

            # scale = (Var+0.0000001).rsqrt()
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
            # scale = 1 / (x_flat.max(-1)[0].max(0)[0] - x_flat.min(-1)[0].min(0)[0])
        # out = x - mean.view(1, mean.size(0), 1, 1)
        # denominator = (var + 0.001).sqrt()

        out = (x - mean.view(1, mean.size(0), 1, 1))* scale.view(1, scale.size(0), 1, 1)

        # out = (x - mean.view(1, mean.size(0), 1, 1)) / \ (scale.view(1, scale.size(0), 1, 1) + 0.001)
        # out = (x - mean.view(1, mean.size(0), 1, 1)/(denominator.view(1, mean.size(0), 1, 1)))

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.scale is not None:
            out = out * self.scale.view(1, self.scale.size(0), 1, 1)

        if self.mean is not None:
            out = out + self.mean.view(1, self.mean.size(0), 1, 1)
        return out













class BatchNorm(nn.Module):
    """docstring for BatchNorm."""

    def __init__(self, num_features, dim=1, affine=True, eps=1e-8):
        super(BatchNorm, self).__init__()
        self.dim = dim
        self.affine = affine
        self.eps = 1e-8
        if affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):
        mean = _mean(x, self.dim).view(1, -1, 1, 1)
        std =  _std(x, self.dim).view(1, -1, 1, 1)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            bias = self.bias.view(1, -1, 1, 1)
            weight = self.weight.view(1, -1, 1, 1)
            x = x * weight + bias
        return x







class MeanBNOld(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):

        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)

        out = x - mean.view(1, mean.size(0), 1, 1)
        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out









class MeanScaleNorm(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True):
        super(MeanScaleNorm, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_scale', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = Parameter(torch.Tensor(num_features))
            self.weight = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):

        if self.training:
            x_flat = x.view(x.size(0), x.size(self.dim), -1)
            mean = x_flat.mean(-1).mean(0)
            scale = 1 / (x_flat.max(-1)[0].max(0)
                         [0] - x_flat.min(-1)[0].min(0)[0])
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))
            self.running_scale.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_scale)
        out = (x - mean.view(1, mean.size(0), 1, 1)) / \
            scale.view(1, scale.size(0), 1, 1)
        if self.weight is not None:
            out = out * self.weight.view(1, self.bias.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class MeanRN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, d_max=1, bias=True):
        super(MeanRN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.d_max = d_max
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):

        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            diff = mean.data - self.running_mean
            d = Variable(diff.clamp(-self.d_max, self.d_max))
            self.running_mean.mul_(self.momentum).add_(
                diff * (1 - self.momentum))
            delta = mean - d
        else:
            delta = torch.autograd.Variable(self.running_mean)
        out = x - delta.view(1, delta.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out
