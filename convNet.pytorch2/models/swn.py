"""
Weight Normalization from https://arxiv.org/abs/1602.07868
taken and adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py
"""
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn


class ShakeShakeNoise(Function):

    @staticmethod
    def forward(ctx, inputs, noise_std):
        output = inputs * inputs.new(inputs.size()).normal_(1, noise_std)
        ctx.noise_std = noise_std
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * \
            Variable(grad_output.data.new(
                grad_output.size()).normal_(1, ctx.noise_std))
        return grad_input, None


def shake_shake_noise(x, noise_std=0.1):
    return ShakeShakeNoise().apply(x, noise_std)


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


class MeanBN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
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


class StochasticWeightNorm(object):

    def __init__(self, name, dim, p, noise_std):
        self.name = name
        self.dim = dim
        self.noise_std = noise_std

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        if module.training and self.noise_std > 0:
            # g = shake_shake_noise(g, self.noise_std)
            g = g * Variable(g.data.new(g.size()).normal_(1, self.noise_std))
        # v = v - _mean(v, self.dim)
        # v = v.renorm(p=2, dim=self.dim, maxnorm=1)
        #return v * g
        return v * (g / _norm(v, self.dim))#.clamp(max=1))
        # return (v - _mean(v, self.dim)) * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, name, dim, p, noise_std):
        fn = StochasticWeightNorm(name, dim, p, noise_std)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + '_g', Parameter(_norm(weight, dim, p=p).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0, p=2, noise_std=0.1):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    StochasticWeightNorm.apply(module, name, dim, p, noise_std)
    return module


def remove_weight_norm(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, StochasticWeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
