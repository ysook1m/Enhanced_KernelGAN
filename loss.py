import torch
import torch.nn as nn
from torch.autograd import Variable
from util import shave_a2b, create_penalty_mask, map2tensor
from torch.nn import functional as F
from scipy.io import loadmat

# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size, conf=None, gantype=None):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        if gantype=='GAN':
            self.loss = nn.BCELoss(reduction='none')
        elif gantype=='LSL1':
            self.loss = nn.L1Loss(reduction='none')
        elif gantype=='LSL2':
            self.loss = nn.L2Loss(reduction='none')
        else:
            print(ddd)
        # Make a shape
        d_last_layer_shape = [conf.bSize, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real, mask=None, lossig=False):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        
        # Compute the loss
        if mask is not None:
            if lossig is True:
                mask = mask * 0.5
            loss = self.loss(d_last_layer*mask, label_tensor*mask)
            
        else:
            loss = self.loss(d_last_layer, label_tensor)
            
            if lossig is True:
                mloss = torch.mean(loss)
                temp1 = loss - mloss
                temp1 = temp1/torch.std(temp1)
                igmask= (0.0 < temp1)*1.0
                print(igmask.size(), torch.sum(igmask))
                igmask= (temp1 < 0.0)*1.0
                print(igmask.size(), torch.sum(igmask))
                igmask= (-0.865 < temp1)*1.0 * (temp1 < 0.865)*1.0
                print(igmask.size(), torch.sum(igmask))
                igmask= (-0.433 < temp1)*1.0 * (temp1 < 1.3)*1.0
                print(igmask.size(), torch.sum(igmask))
                igmask= (-1.3 < temp1)*1.0 * (temp1 < 0.433)*1.0
                print(igmask.size(), torch.sum(igmask))
                
                loss = loss * igmask.detach()
        
        loss = torch.mean(loss)
        return loss

class GTLoss(nn.Module): #greater than
    def __init__(self, d_last_layer_size, conf=None):
        super(GTLoss, self).__init__()
        
        self.loss = nn.L1Loss(reduction='mean')
        
        d_last_layer_shape = [conf.bSize, 1, d_last_layer_size, d_last_layer_size]
        
        self.label_zeros = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_zeros18 = Variable(torch.zeros(conf.bSize, 1, d_last_layer_size-2, d_last_layer_size-2).cuda(), requires_grad=False)
    
    def forward(self, lessFake, moreFake, margin, mask=None, size18=False):
        loss_sub1 = F.relu(moreFake-lessFake + 0.1)
        loss_sub2 = F.relu((moreFake+1e-10)/(lessFake+1e-10) - margin)
        loss_sub  = torch.max(loss_sub1, loss_sub2)
        
        if size18 is False:
            label = self.label_zeros
        else:
            label = self.label_zeros18
            
        if mask is not None:
            loss = self.loss(loss_sub*mask, label*mask)
        else:
            loss = self.loss(loss_sub, label)
        
        return loss


class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor, conf=None):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bk = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.bk = self.bk.expand(1, 1, self.bk.size(0), self.bk.size(1))
        self.bk_p = (self.bk.size(-1) - 1) // 2
        self.bSize = conf.bSize
        self.hwSize = conf.input_crop_size

    def forward(self, g_in, g_out):
        g_in = g_in.view(-1, 1, self.hwSize, self.hwSize).contiguous()
        downscaled = F.conv2d(g_in, self.bk, stride=2, padding=self.bk_p)
        downscaled = downscaled.view(self.bSize, -1, self.hwSize//2, self.hwSize//2).contiguous()
        
        return self.loss(g_out, shave_a2b(downscaled, g_out))


class DownScaleBicSub(nn.Module):

    def __init__(self, scale_factor, conf=None):
        super(DownScaleBicSub, self).__init__()
        ####################################################################
        matmat = loadmat('matfiles/gauss60.mat')
        gauss60 = matmat['val']
        ####################################################################
        self.avg2 = Variable(torch.Tensor(gauss60).cuda(), requires_grad=False)
        self.avg2 = self.avg2.unsqueeze(0).unsqueeze(0)
        self.pad2 = self.avg2.size(-1)//2
        
        self.st = round(1 / scale_factor) # scale_factor == 0.5

    def forward(self, g_in):
        b,c,h,w = g_in.size()
        o_sub2 = g_in.view(-1, 1, h, w).contiguous()
        o_sub2 = F.conv2d(o_sub2, self.avg2, stride=self.st, padding=self.pad2) #1,3,32,32
        o_sub2 = o_sub2.view(-1, c, h//self.st, w//self.st).contiguous()
        shv = (32-26)//2 # 32x32 => 26x26
        o_sub2 = o_sub2[:,:,shv:-shv, shv:-shv]
        return o_sub2
        
        
class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self, conf=None):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.label = torch.tensor(1.0).cuda()

    def forward(self, kernel):
        return self.loss(self.label, torch.sum(kernel))

        
class ConcaveLoss(nn.Module):
    """ Penalizes concave"""
    def __init__(self):
        super(ConcaveLoss, self).__init__()
        ####################################################################
        matmat = loadmat('matfiles/mask.mat')
        maskA = matmat['maskA']
        maskB = matmat['maskB']
        nA = matmat['nA']
        ####################################################################
        self.maskA = Variable(torch.Tensor(maskA).cuda(), requires_grad=False)
        self.maskB = Variable(torch.Tensor(maskB).cuda(), requires_grad=False)
        self.nA    = Variable(torch.Tensor(nA).cuda(), requires_grad=False)
        self.rateMax = 0.5
        # self.zer = Variable(torch.zeros(1).cuda(), requires_grad=False)

    def forward(self, kernel):
        nmask = self.maskA.size(-1)# 9
        concSum = 0
        for i in range(nmask): #0,1,2,3,4,5,6,7,8
            maskA = self.maskA[:,:,i]
            maskB = self.maskB[:,:,i]
            
            aMean = torch.sum(kernel*maskA)/self.nA[:,i]
            aMax = torch.max(kernel*maskA)
            aUpper = (self.rateMax*aMax + (1.0-self.rateMax)*aMean)*0.9
            concSum = concSum + torch.sum(F.relu((kernel - aUpper)*maskB))
            
        return concSum

class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5, conf=None):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2# + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False).unsqueeze(-1)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        kernel0 = kernel - torch.min(kernel).detach()
        r_sum, c_sum = torch.sum(kernel0, dim=1).reshape(1, -1), torch.sum(kernel0, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel0.detach()),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel0.detach()))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size, conf=None):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(1,1,k_size,k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self, conf=None):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
