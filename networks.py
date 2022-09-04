import torch
import torch.nn as nn
from util import swap_axis
from torch.autograd import Variable
from torch.nn import functional as F
import random

class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        self.sf = conf.scale_factor
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False, padding=(struct[0]-1)//2)
        
        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False, padding=(struct[layer]-1)//2)]
        self.feature_block = nn.Sequential(*feature_block)
        
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], bias=False, padding=(struct[-1]-1)//2)
        
        self.hwSize = conf.input_crop_size
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size])))[0].shape[-1] #32
        

    def forward(self, in1):
        b,c,h,w = in1.size()
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        out1 = in1.view(-1, 1, h, w).contiguous()
        out1 = self.first_layer(out1)
        out1 = self.feature_block(out1)
        out1 = self.final_layer(out1)
        
        out1 = F.avg_pool2d(out1, kernel_size=2, stride=2, padding=0)
        out1 = out1.view(-1, 3, h//2, w//2).contiguous()
        shv = (32-26)//2
        out1 = out1[:,:,shv:-shv, shv:-shv] # 16,3,26,26
        return out1


class Discriminator(nn.Module):

    def __init__(self, conf):
        super(Discriminator, self).__init__()
        self.dd = conf.D_chan
        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        self.first_layer2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size-2, bias=True))
        self.first_layer3 = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size-4, bias=True))
        
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        
        feature_block2 = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block2 += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_block2 = nn.Sequential(*feature_block2)
        
        feature_block3 = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block3 += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_block3 = nn.Sequential(*feature_block3)
        
        
        
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)), nn.Sigmoid())
        
        
        # First layer - Convolution (with no ReLU)
        self.first_layerb = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        self.first_layer2b = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size-2, bias=True))
        self.first_layer3b = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size-4, bias=True))
        
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_blockb = nn.Sequential(*feature_block)
        
        feature_block2 = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block2 += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_block2b = nn.Sequential(*feature_block2)
        
        feature_block3 = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block3 += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)), nn.BatchNorm2d(conf.D_chan), nn.ReLU(True)]
        self.feature_block3b = nn.Sequential(*feature_block3)
        
        
        
        self.final_layerb = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)), nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.ec = conf.D_kernel_size//2
        self.dPosEnc = False#conf.dPosEnc
  
    def forward(self, input_tensor, in_enc=None):
        
        if (in_enc is not None) and (self.dPosEnc==True):
            print(in_enc.size())
            size    = self.dd//in_enc.size(1) #64/4
            in_enc  = in_enc.repeat(1,size,1,1).contiguous()
        ###########################################
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        
        receptive_extraction2 = self.first_layer2(input_tensor)
        features2 = self.feature_block2(receptive_extraction2)
        
        receptive_extraction3 = self.first_layer3(input_tensor)
        features3 = self.feature_block3(receptive_extraction3)
        
        features = features + features2[:,:,1:-1,1:-1] + features3[:,:,2:-2,2:-2]
        
        if (in_enc is not None) and (self.dPosEnc==True):
            features = features * in_enc[:,:,self.ec:-self.ec,self.ec:-self.ec]
        
        featuresa = self.final_layer(features)
        
        
        
        
        
        if (in_enc is not None) and (self.dPosEnc==True):
            print(in_enc.size())
            size    = self.dd//in_enc.size(1) #64/4
            in_enc  = in_enc.repeat(1,size,1,1).contiguous()
        ###########################################
        receptive_extractionb = self.first_layerb(input_tensor)
        featuresb = self.feature_blockb(receptive_extractionb)
        
        receptive_extraction2b = self.first_layer2b(input_tensor)
        features2b = self.feature_block2b(receptive_extraction2b)
        
        receptive_extraction3b = self.first_layer3b(input_tensor)
        features3b = self.feature_block3b(receptive_extraction3b)
        
        featuresb = featuresb + features2b[:,:,1:-1,1:-1] + features3b[:,:,2:-2,2:-2]
        
        if (in_enc is not None) and (self.dPosEnc==True):
            featuresb = featuresb * in_enc[:,:,self.ec:-self.ec,self.ec:-self.ec]
        
        featuresb = self.final_layerb(featuresb)
        
        out = torch.cat([featuresa.unsqueeze(4), featuresb.unsqueeze(4)], dim=4)
        input_max, input_indexes = torch.max(out, dim=4)
        
        return input_max


def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        
        size=m.weight.size(-1)
        ch=m.weight.size(0)
        
        K = torch.zeros(m.weight.size())
        K[:,:,size//2,size//2] = 0.1/ch
        with torch.no_grad():
            m.weight=torch.nn.Parameter(K.cuda() + m.weight)
        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
