import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np



class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(self, x):
        q = 512
        self.save_for_backward(x)
        x_q = x.mul_(q).round_().div_(q)
        return x_q


    def backward(self, grad_output):
        q = 512
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input.mul_(q).round_().div_(q)
        return grad_input


#Quantized L1
class QuantizedL1(nn.Module):
    # This is normalized L1 Batch norm
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
        print('L1')

    def forward(self, x):
        p=1
        if self.training:
            x.half()
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            # x = x.half()
            # mean = mean.half()

            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)


            # quant = Quantize.apply
            # Var1 = quant(torch.abs((t.transpose(1, 0) - mean)))
            # Var = quant(Var1.mean(0))
            # scale = quant(((Var * (np.pi / 2) ** 0.5) + 0.0000001) ** (-1))

            Var1 = torch.abs((t.transpose(1, 0) - mean))

            # Var1 = Var1.float()
            Var = Var1.mean(0)
            # Var = Var.half()

            scale = ((Var * (np.pi / 2) ** 0.5) + 0.0000001) ** (-1)

            scale = scale.float()
            mean = mean.float()
            x = x.float()

            # Mean = mean.data.type(torch.cuda.FloatTensor)
            # Scale = scale.data.type(torch.cuda.FloatTensor)
            # self.running_mean.mul_(self.momentum).add_(
            #     Mean * (1 - self.momentum))
            #
            # self.running_var.mul_(self.momentum).add_(
            #     Scale * (1 - self.momentum))


            #####old

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
            ######





        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1))  * scale.view(1, scale.size(0), 1, 1)

        # out = (x - mean.view(1, mean.size(0), 1, 1)) / \ (scale.view(1, scale.size(0), 1, 1) + 0.001)
        # out = (x - mean.view(1, mean.size(0), 1, 1)/(denominator.view(1, mean.size(0), 1, 1)))

        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise

        if self.scale is not None:
            #old
            out = out * self.scale.view(1, self.scale.size(0), 1, 1)
            # old

            # new
            # ScaleLearnable = self.scale.half()
            # out = out * ScaleLearnable.view(1, ScaleLearnable.size(0), 1, 1)
            # # end new


        if self.mean is not None:
            #old
            out = out + self.mean.view(1, self.mean.size(0), 1, 1)
            # old

            # new
            # MeanLearnable = self.mean.half()
            # out = out + MeanLearnable.view(1, MeanLearnable.size(0), 1, 1)
            # # end new


        return out




#Quantized_L2
class Ql2(nn.Module):
    #This is L2 Baseline

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.mean = Parameter(torch.Tensor(num_features))
        self.scale = Parameter(torch.Tensor(num_features))
        print('L2')

    def forward(self, x):
        p=2
        if self.training:
            x = x.half()
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            # x = x.half()
            # mean = mean.half()

            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            # Var = (torch.abs((t.transpose(1, 0) - mean)) ** p).mean(0)

            # quant = Quantize.apply
            # Var1 = quant(torch.abs((t.transpose(1,0)-mean)))
            # Var2 = quant(Var1 ** p)
            # Var =  quant(Var2.mean(0))
            # scale = quant((Var + 0.0000001)**(-1/p))



            Var1 = torch.abs((t.transpose(1, 0) - mean))
            Var2 = Var1 ** p

            # Var2 = Var2.float()
            Var = Var2.mean(0)
            # Var = Var.half()

            scale = (Var + 0.0000001) ** (-1 / p)

            scale = scale.float()
            mean = mean.float()
            x = x.float()



            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1))* scale.view(1, scale.size(0), 1, 1)

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

#L2
class L2(nn.Module):
    #This is L2 Baseline

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
        p=2
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)

            import time
            # torch.cuda.synchronize()
            # a = time.perf_counter()
            Var = (torch.abs((t.transpose(1,0)-mean))**p).mean(0)
            scale = (Var + 0.0000001)**(-1/p)
            # b = time.perf_counter()
            # print('L2 {:.02e}s'.format(b - a))

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

        out = (x - mean.view(1, mean.size(0), 1, 1))* scale.view(1, scale.size(0), 1, 1)

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


#Top1
class Top1(nn.Module):
    # this is normalized L_inf

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
        K = 1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)

            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / ((2 * np.log(A.size(0))) ** 0.5)

            MeanTOPK = (torch.topk(A, K, dim=0)[0].mean(0))*const
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

#Top10
class Top10(nn.Module):
    ## This is normalized Top10 batch norm


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


    def forward(self, x):
        # p=5
        K=10
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            A = torch.abs(t.transpose(1, 0) - mean)
            beta = 0.75
            

            MeanTOPK = torch.topk(A,K,dim=0)[0].mean(0)
            meanTOPK = beta*torch.autograd.variable.Variable(self.meanTOPK) + (1-beta)*MeanTOPK

            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / ((2 * np.log(A.size(0))) ** 0.5)
            meanTOPK = meanTOPK * const
            
            # print(self.meanTOPK)
            self.meanTOPK.copy_(meanTOPK.data)
            # self.meanTOPK = MeanTOPK.data
            scale = 1/(meanTOPK + 0.0000001)

            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

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


#L1
class MeanBN(nn.Module):
    # This is normalized L1 Batch norm
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

        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)

            # import time
            # torch.cuda.synchronize()
            # a = time.perf_counter()
            Var = (torch.abs((t.transpose(1,0)-mean))).mean(0)
            # Var = (torch.abs((t.transpose(1, 0) - mean)) ** p).mean(0)
            scale = ((Var * (np.pi / 2) ** 0.5) + 0.0000001) ** (-1)
            # b = time.perf_counter()
            # print('L1 {:.02e}s'.format(b - a))


            # scale = ((Var * (np.pi / 2) ** 0.5) + 0.0000001) ** (-1 / p)
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

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

# L1 - unormalized
class unormalized(nn.Module):
    # This is normalized L1 Unormalized Batch norm
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
        p=1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0,1)
            z = y.contiguous()
            t = z.view(z.size(0),-1)
            Var = (torch.abs((t.transpose(1,0)-mean))**p).mean(0)
            scale = (Var  + 0.0000001) ** (-1 / p)            
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)

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


#Range (max - min)
class Range(nn.Module):
    # this is normalized L_inf

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
        K = 1
        NumOfChunks = 10
        C = 0
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = t.transpose(1, 0) - mean

            B = torch.chunk(A, NumOfChunks, dim=0)
            # B = A.view(NumOfChunks, A.size(0) // NumOfChunks, *(list(A.size())[2:]))

            for i in range(0,NumOfChunks):
                const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5) / ((2 * np.log(B[i].size(0))) ** 0.5)
                C = C + (torch.max(B[i], dim=0)[0] - torch.min(B[i], dim=0)[0])* const

            MeanTOPK = C/NumOfChunks

            # MAX = torch.max(A,dim = 0)[0]
            # MIN = torch.min(A,dim = 0)[0]
            # MeanTOPK = (MAX - MIN) * const

            # MeanTOPK = (torch.topk(A, K, dim=0)[0].mean(0))*const

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
