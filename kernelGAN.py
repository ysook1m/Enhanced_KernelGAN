import torch
import loss
import networks
import torch.nn.functional as F
from util import save_kernel, run_zssr, post_process_k, post_proc, load_kernel, move2cpu, centralized_k, significant_k, save_kernel_nonc, signif_k
import numpy as np
import scipy.io as sio
import os
from scipy.io import loadmat
from torch.autograd import Variable
import matplotlib.pyplot as plt
from DIPnetworks import Predictor, skip, fcn
from SSIM import SSIM
from DIPutil import save_final_kernel_png, get_noise, kernel_shift, tensor2im01
from collections import OrderedDict



class KernelGAN:
    ##### Constraint co-efficients
    lmd_sum2one = 0.5
    lmd_bicubic = 5
    lmd_concave = 5.0
    
    lmd_boundaries = 5.0
    lambda_centralized = 0
    lambda_sparse = 0
    lmd_sym = 1.0
    lmd_u0 = 1.0
    
    lmd_fakeblur = 0.5
    cnt_iter = 0
    def __init__(self, conf):
        ##### Acquire configuration
        self.rankDisc   = conf.rankDisc #True False
        self.genMask    = conf.genMask #True False
        self.dscMask    = conf.dscMask #True False
        self.gantype    = conf.gantype #GAN LSL1 LSL2
        self.wSTD       = conf.wSTD
        self.conf       = conf
        print('gantype:', self.gantype, ',  wSTD:', self.wSTD, ',  rankDisc:', self.rankDisc, ',  genMask:', self.genMask, ',  dscMask:', self.dscMask, ',  dVar:', conf.dVar)
        
        ##### Define the GAN
        self.G  = networks.Generator(conf).cuda()
        self.D2 = networks.Discriminator(conf).cuda()
        
        ##### Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D2.forward_shave
        
        ##### Input tensors
        # self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        # self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()
        # self.g_inputB= torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        # self.d_inputB= torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()
        # self.mg_inB  = torch.FloatTensor(1, 1, 20, 20).cuda()
        # self.md_inB  = torch.FloatTensor(1, 1, 20, 20).cuda()
        # self.g_input = -1
        # self.d_input = -1
        # self.g_inputB= -1
        # self.d_inputB= -1
        # self.mg_inB  = -1
        # self.md_inB  = -1
        
        ##### The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()
        
        ##### Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=self.d_output_shape, conf=conf, gantype=self.gantype).cuda()
        self.bicubic_loss = loss.DownScaleLoss(scale_factor=conf.scale_factor, conf=conf).cuda()
        self.sum2one_loss = loss.SumOfWeightsLoss(conf=conf).cuda()
        self.boundaries_loss = loss.BoundariesLoss(k_size=conf.G_kernel_size, conf=conf).cuda()
        self.centralized_loss = loss.CentralizedLoss(k_size=conf.G_kernel_size, scale_factor=conf.scale_factor, conf=conf).cuda()
        self.sparse_loss = loss.SparsityLoss(conf=conf).cuda()
        self.l_bicubic = 0
        
        ##### downsampling by Bic, Sub
        self.ConcaveLoss = loss.ConcaveLoss().cuda()
        self.DownScaleBicSub = loss.DownScaleBicSub(scale_factor=conf.scale_factor, conf=conf).cuda()
        self.GT_loss_layer = loss.GTLoss(d_last_layer_size=self.d_output_shape, conf=conf).cuda()
        self.criterionGT = self.GT_loss_layer.forward
        
        ##### Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward
        
        ##### Initialize networks weights
        self.G.apply(networks.weights_init_G)
        self.D2.apply(networks.weights_init_D)
        
        ##### Optimizers
        self.optimizer_K = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))
        
        self.l1_g = np.zeros(conf.max_iters)
        self.l2_g = np.zeros(conf.max_iters)
        self.loss_cons = np.zeros(conf.max_iters)
        self.lc1 = np.zeros(conf.max_iters)
        self.lc2 = np.zeros(conf.max_iters)
        self.lc3 = np.zeros(conf.max_iters)
        self.lc4 = np.zeros(conf.max_iters)
        self.lc5 = np.zeros(conf.max_iters)
        
        self.l_d1_fake = np.zeros(conf.max_iters)
        self.l_d1_real = np.zeros(conf.max_iters)
        self.l_d1_bic = np.zeros(conf.max_iters)
        
        self.l_d2_fake = np.zeros(conf.max_iters)
        self.l_d2_real = np.zeros(conf.max_iters)
        self.l_d2_sub = np.zeros(conf.max_iters)
        
        self.Ldx = 0
        
        self.sub_dif = torch.zeros([1024, 1024])
        self.bic_dif = torch.zeros([1024, 1024])
        self.mask_dif= torch.zeros([1024, 1024])
        
        
        #############################################################
        self.v_center=conf.vCenter
        print('3x3 bk center value:', self.v_center)
        self.bk3x3 = Variable(torch.rand(conf.max_iters, conf.bSize, 3,3).cuda(), requires_grad=False)
        self.bk3x3[:,:,1,1]=0
        bk_sum=torch.sum(self.bk3x3, dim=(2,3), keepdim=True)
        self.bk3x3 = self.bk3x3/bk_sum*(1.0-self.v_center)
        self.bk3x3[:,:,1,1]=self.v_center
        #############################################################
        self.corrector = Predictor().cuda()
        load_path = './matfiles/latest_G.pth'
        
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        self.corrector.load_state_dict(load_net_clean, strict=True)
        for param in self.corrector.parameters():
            param.requires_grad = False
        #############################################################
        if not os.path.isdir(os.path.join(self.conf.output_dir_path, 'kernel')):
            os.makedirs(os.path.join(self.conf.output_dir_path, 'kernel'))
        if not os.path.isdir(os.path.join(self.conf.output_dir_path, 'png_sr')):
            os.makedirs(os.path.join(self.conf.output_dir_path, 'png_sr'))
        if not os.path.isdir(os.path.join(self.conf.output_dir_path, 'png_kernel')):
            os.makedirs(os.path.join(self.conf.output_dir_path, 'png_kernel'))
        gtpath = '../data/DIV2KRK/gt_k_x2/kernel_' + conf.img_name.split('_')[-1] + '.mat'
        
        self.gt_kernel = sio.loadmat(gtpath)['Kernel']
        self.gt_kernel = np.pad(self.gt_kernel, 1)
        #############################################################
        print('*' * 10 + '\nSTARTED KernelGAN on: \"%s\"...' % conf.input_image_path)
        
    def train(self, g_input, d_input, g_inputB, d_inputB, mg_inB, md_inB, conf):

        self.g_input    = g_input.contiguous()
        self.d_input    = d_input.contiguous()
        self.train_d(conf)
        self.train_g(conf)
        #############################################################
        self.calc_kernel()
        
        # freeze kernel estimation, so that DIP can train first to learn a meaningful image
        if conf.iter <= 500:
            corrected_kernel = self.corrector(self.curr_k.unsqueeze(0).unsqueeze(0)).detach()
        else:
            corrected_kernel = self.corrector(self.curr_k.unsqueeze(0).unsqueeze(0))
        corrected_kernel = corrected_kernel/corrected_kernel.sum()
        self.optimizer_dip.zero_grad()
        self.ksize = 13
        '''
        # (2.1) forward
         '''
        # generate sr image
        sr      = self.net_dip(self.input_dip)
        sr_pad  = F.pad(sr, mode='circular', pad=(self.ksize // 2, self.ksize // 2, self.ksize // 2, self.ksize // 2))
        out     = F.conv2d(sr_pad, corrected_kernel.expand(3, -1, -1, -1), groups=3)
        
        # downscale
        # out = out[:, :, 0::2, 0::2]
        out = F.avg_pool2d(out, kernel_size=2, stride=2, padding=0)
        
        '''
        # (2.2) backward
         '''
        # first use SSIM because it helps the model converge faster
        if conf.iter <= 550:
            loss = 1 - self.ssimloss(out, self.lr)
        else:
            loss = self.mse(out, self.lr)
        
        loss.backward()
        self.optimizer_dip.step()
        if conf.iter%100 == 0 or conf.iter==(conf.max_iters-1):
            print('\n Iter {}, loss: {}'.format(conf.iter, loss.data))
            save_final_kernel_png(move2cpu(corrected_kernel.squeeze()), self.conf, self.gt_kernel, conf.iter)
            plt.imsave(os.path.join(self.conf.output_dir_path, 'png_sr/{}_{}.png'.format(self.conf.img_name, conf.iter)),
                                tensor2im01(sr), vmin=0, vmax=1., dpi=1)
        
        self.Ldx = self.Ldx + 1
        
    def dip_input(self, input):
        input = torch.FloatTensor(np.transpose(input, (2, 0, 1))).cuda().unsqueeze(0)
        crop = int(960 / 2 / 2)
        self.lr = input[:, :, input.shape[2] // 2 - crop: input.shape[2] // 2 + crop,
                       input.shape[3] // 2 - crop: input.shape[3] // 2 + crop]
                       
        # DIP model
        _, C, H, W = self.lr.size()
        self.input_dip = get_noise(C, 'noise', (H * 2, W * 2)).cuda().detach()
        
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.cuda()
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=2e-2)
        
        # initialze the kernel to be smooth is slightly better
        seed = 5
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        self.ssimloss = SSIM().cuda()
        self.mse = torch.nn.MSELoss().cuda()
        
        
        
        
        
        #############################################################
                       
                       
                       
                       
    def log_save(self):
        sio.savemat(os.path.join(self.conf.output_dir_path, '%s_log.mat' % self.conf.img_name), {\
        'l1_g'          : self.l1_g   , \
        'l2_g'          : self.l2_g  , \
        'loss_cons'     : self.loss_cons, \
        'lc1'           : self.lc1      , \
        'lc2'           : self.lc2      , \
        'lc3'           : self.lc3      , \
        'lc4'           : self.lc4      , \
        'lc5'           : self.lc5      , \
        'l_d1_fake'     : self.l_d1_fake     , \
        'l_d1_real'     : self.l_d1_real     , \
        'l_d1_bic'      : self.l_d1_bic      , \
        'l_d2_fake'     : self.l_d2_fake     , \
        'l_d2_real'     : self.l_d2_real     , \
        'l_d2_sub'      : self.l_d2_sub})

    ##### noinspection PyUnboundLocalVariable
    def calc_kernel(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.first_layer.parameters()):
            curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) #1, 64, 19, 19]
        for ind, w in enumerate(self.G.feature_block.parameters()):
            curr_k = F.conv2d(curr_k, w)                                     #[1, 64, 15, 15][1, 64, 13, 13]
        for ind, w in enumerate(self.G.final_layer.parameters()):
            curr_k = F.conv2d(curr_k, w)                                     #[1, 1, 13, 13]
        curr_k = curr_k.squeeze().flip([0, 1])                          #[13, 13]
        
        self.curr_k = curr_k
        

        
    def l_symmetric(self):
        kk = self.curr_k.flip([0, 1])
        return torch.sum(torch.abs(kk - self.curr_k))*0.5

    def l_up0(self):
        kk = self.curr_k*-1.0
        return torch.sum(F.relu(kk))
        
    def calc_constraints(self, g_in, g_pred):
        ##### Calculate K which is equivalent to G
        self.calc_kernel()
        ##### Calculate constraints
        self.l_bicubic  = self.bicubic_loss.forward(g_in=g_in, g_out=g_pred)
        l_sum2one       = self.sum2one_loss.forward(kernel=self.curr_k)
        l_concave       = self.ConcaveLoss.forward(kernel=self.curr_k)
        # l_sym           = self.l_symmetric()
        l_boundaries    = self.boundaries_loss.forward(kernel=self.curr_k)
        l_u0 = self.l_up0()
        
        ##### Apply constraints co-efficients
        return self.l_bicubic * self.lmd_bicubic + l_sum2one * self.lmd_sum2one + l_concave * self.lmd_concave + l_u0 * self.lmd_u0 + l_boundaries * self.lmd_boundaries, \
        self.l_bicubic, l_sum2one, l_concave, None, l_u0

    def initialize_kernel(self):
        matmat = loadmat('matfiles/init_kernel.mat')
        init_ker = matmat['val']
        init_ker = Variable(torch.Tensor(init_ker).cuda(), requires_grad=False)
        init_ker = init_ker/torch.sum(init_ker)
        for iter_k in range(100):
            self.optimizer_K.zero_grad()
            self.calc_kernel()
            loss = torch.sum(torch.abs(init_ker-self.curr_k))
            loss.backward()
            self.optimizer_K.step()
        
        with torch.no_grad():
            layer_parameters = (self.G.first_layer.parameters(), self.G.feature_block.parameters(), self.G.final_layer.parameters())
            for layer in layer_parameters:
                for ind, w in enumerate(layer): #1, 64, 19, 19]
                    w.add_(torch.randn(w.size()).cuda() * 0.01)
        
    def train_g(self, conf):
        self.optimizer_G.zero_grad()
        if self.genMask==False: #1001 1111
            g_pred = self.G.forward(self.g_input)
            d2_pred_fake = self.D2.forward(g_pred)
            if conf.iter<1000:
                l2_g = self.criterionGAN(d_last_layer=d2_pred_fake, is_d_input_real=True, mask=self.wSTD)
            else:
                l2_g = self.criterionGAN(d_last_layer=d2_pred_fake, is_d_input_real=True, mask=self.wSTD, lossig=True)
                
            loss_cons, lc1, lc2, lc3, lc4, lc5 = self.calc_constraints(g_in=self.g_input, g_pred=g_pred)
        elif self.genMask==True: #1001 1111
            g_pred = self.G.forward(self.g_inputB)
            d2_pred_fake = self.D2.forward(g_pred)
            l2_g = self.criterionGAN(d_last_layer=d2_pred_fake, is_d_input_real=True, mask=self.mg_inB*self.wSTD, lossig=True)
            loss_cons, lc1, lc2, lc3, lc4, lc5 = self.calc_constraints(g_in=self.g_inputB, g_pred=g_pred)
        else:
            print(ddd)
        
        ##### Sum all losses
        total_l_g = l2_g + loss_cons
        
        total_l_g.backward()
        self.optimizer_G.step()
        
        self.l2_g[self.Ldx] = l2_g.item()
        self.loss_cons[self.Ldx] = loss_cons.item()
        self.lc1[self.Ldx] = lc1.item()
        self.lc2[self.Ldx] = lc2.item()
        self.lc3[self.Ldx] = lc3.item()
        self.lc5[self.Ldx] = lc5.item()
        
    def train_d(self, conf):
        self.optimizer_D2.zero_grad()
        if self.dscMask==False: #1001 1111
            g_pred          = self.G.forward(self.g_input)
            d2_pred_fake    = self.D2.forward((g_pred + torch.randn_like(g_pred) / 255.).detach())
            l_d2_fake       = self.criterionGAN(d2_pred_fake, is_d_input_real=False)
        elif self.dscMask==True: #1001 1111
            g_pred          = self.G.forward(self.g_inputB)
            d2_pred_fake    = self.D2.forward((g_pred + torch.randn_like(g_pred) / 255.).detach())
            l_d2_fake       = self.criterionGAN(d2_pred_fake, is_d_input_real=False, mask=self.mg_inB)
        else:
            print(ddd)
        
        if self.dscMask==False: #1001 1111
            d2_pred_real  = self.D2.forward(self.d_input)
            l_d2_real     = self.criterionGAN(d2_pred_real, is_d_input_real=True)
        elif self.dscMask==True: #1001 1111
            d2_pred_real  = self.D2.forward(self.d_inputB)
            l_d2_real     = self.criterionGAN(d2_pred_real, is_d_input_real=True, mask=self.md_inB)
        else:
            print(ddd)
        
        ##################################################
        if self.rankDisc==True: #True False
            if self.dscMask==False: #1001 1111
                d_sub2        = self.DownScaleBicSub(self.g_input)
                d2_sub_fake2  = self.D2.forward((d_sub2 + torch.randn_like(d_sub2) / 255.).detach())
                l_d2_sub      = self.criterionGT(d2_pred_fake, d2_sub_fake2, 0.8) #lessFake, moreFake
                
                
                #########################
                real_in       = self.d_input.permute(1,0,2,3).contiguous()
                bk33          = self.bk3x3[conf.iter, :, :,:].unsqueeze(1)
                realfake      =  F.conv2d(real_in, bk33, stride=1, padding=0, groups=conf.bSize)
                realfake      =  realfake.permute(1,0,2,3).contiguous().detach()
                
                d2_realfake   = self.D2.forward(realfake)
                l_realfake    = self.criterionGT(d2_pred_real[:,:,1:-1,1:-1], d2_realfake, 0.8, size18=True) #lessFake, moreFake
                #########################
            elif self.dscMask==True: #1001 1111
                d_sub2        = self.DownScaleBicSub(self.g_inputB)
                d2_sub_fake2  = self.D2.forward((d_sub2 + torch.randn_like(d_sub2) / 255.).detach())
                l_d2_sub      = self.criterionGT(d2_pred_fake, d2_sub_fake2, 0.8, mask=self.mg_inB) #lessFake, moreFake
            else:
                print(ddd)
            l_d2          = (l_d2_fake + l_d2_real + l_d2_sub + l_realfake) * 0.5
        elif self.rankDisc==False:
            l_d2          = (l_d2_fake + l_d2_real) * 0.5
        else:
            print(ddd)
        
        l_d2.backward()
        self.optimizer_D2.step()
        
        ##################################################
        self.l_d2_fake[self.Ldx]      = l_d2_fake.item()
        self.l_d2_real[self.Ldx]      = l_d2_real.item()
        # self.l_d2_sub[self.Ldx]       = l_d2_sub.item()
        

        
    def do_ZSSR(self):
        for i in self.conf.kerList:
            k_2, k_4 = load_kernel(self.conf, str(i))
            print('ZSSR start!')
            run_zssr(k_2, k_4, self.conf, str(i))
            print('FINISHED RUN (see --%s-- folder)\n' % self.conf.output_dir_path + '*' * 60 + '\n\n')
        
    def finish(self):
        self.calc_kernel()
        save_kernel_nonc(move2cpu(self.curr_k), self.conf, '777')
        for i in self.conf.maxPostp:
            final_kernel = significant_k(self.curr_k, n=i)
            save_kernel_nonc(final_kernel, self.conf, str(i))
        for i in self.conf.ratPostp:
            final_kernel = signif_k(self.curr_k, n=i)
            save_kernel_nonc(final_kernel, self.conf, str(i))
            
        
        # self.calc_kernel_final()
        # final_kernel = centralized_k(self.curr_k)
        # save_kernel(final_kernel, self.conf, '777')
        # for i in self.conf.maxPostp:
            # final_kernel = post_process_k(self.curr_k, n=i)
            # save_kernel(final_kernel, self.conf, str(i))
        # for i in self.conf.ratPostp:
            # final_kernel = post_proc(self.curr_k, n=i)
            # save_kernel(final_kernel, self.conf, str(i))
            
        print('KernelGAN estimation complete!')
