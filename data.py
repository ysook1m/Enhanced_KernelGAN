import torch
import numpy as np
from torch.utils.data import Dataset
from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F
import torch.nn as nn
import cv2

class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def __init__(self, conf, gan):
        # Default shapes
        self.bSize = conf.bSize
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - gan.D2.forward_shave
        ##preprocessing #######################
        self.input_image = read_image(conf.input_image_path) / 255. #h,w,c
        self.in_dip = self.input_image
        self.input_imageB= read_image(conf.input_image_path) / 255. #h,w,c
        
        if conf.dVar==777:
            print('dvar=777')
            self.input_image = (self.input_image - np.mean(self.input_image, axis=(0,1), keepdims=True))
            self.input_image = (self.input_image/np.std(self.input_image, axis=(0,1), keepdims=True))
            self.input_imageB = (self.input_imageB - np.mean(self.input_imageB, axis=(0,1), keepdims=True))
            self.input_imageB = (self.input_imageB/np.std(self.input_imageB, axis=(0,1), keepdims=True))
        else:
            # self.input_image    = (self.input_image - 0.1)
            self.input_image    = (self.input_image*conf.dVar)
            # self.input_imageB   = (self.input_imageB - 0.1)
            self.input_imageB   = (self.input_imageB*conf.dVar)
        print('input minmax', self.input_image.min(), self.input_image.max())
        
        fname = '../edge_extraction/synthe_img/MaskD'+str(conf.g_thc)+'_use.png'
        self.maskB = (read_image(fname)>0)*1.0
        fname = '../edge_extraction/synthe_img/MaskD'+str(conf.d_thc)+'_use.png'
        self.maskBd = (read_image(fname)>0)*1.0
        #####################################
        
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)
        self.in_rows, self.in_cols = self.input_image.shape[0:2]
        
        # Create prob map for choosing the crop
        self.crop_indices_for_g, self.crop_indices_for_d  = self.make_list_of_crop_indices(conf=conf)
        self.crop_indices_for_gB,self.crop_indices_for_dB = self.make_list_of_crop_indicesB(conf=conf)
        """
        # mean, std on unmasked area with sampling
        #########################################################################
        maxSample = 10000 #3000*16
        g_sample = torch.ones(maxSample, 3, 26, 26).cuda()*1000000.0
        d_sample = torch.ones(maxSample, 3, 7, 7).cuda()*1000000.0
        g_sampleB= torch.ones(maxSample, 3, 26, 26).cuda()*1000000.0
        d_sampleB= torch.ones(maxSample, 3, 7, 7).cuda()*1000000.0
        for iii in range(maxSample):
            g_sample[iii,:,:,:], _, _       = self.next_crop(for_g=True, idx=iii, cropsize=26) #size 64 (for_g=True ==> select in small map: receptive field is large)
            d_sample[iii,:,:,:], _, _       = self.next_crop(for_g=False, idx=iii, cropsize=7) #size 26
            g_sampleB[iii,:,:,:], _, _, _   = self.next_crop_mask(for_g=True, idx=iii, cropsize=26) #size 64 (for_g=True ==> select in small map: receptive field is large)
            d_sampleB[iii,:,:,:], _, _, _   = self.next_crop_mask(for_g=False, idx=iii, cropsize=7) #size 26
        g_meanA     = torch.mean(g_sample, dim=(0,2,3), keepdim=True)
        g_sample    = (g_sample - g_meanA)
        g_stdA      = torch.std(g_sample, dim=(0,2,3), keepdim=True)
        
        d_meanA     = torch.mean(d_sample, dim=(0,2,3), keepdim=True)
        d_sample    = (d_sample - d_meanA)
        d_stdA      = torch.std(d_sample, dim=(0,2,3), keepdim=True)
        
        g_meanB     = torch.mean(g_sampleB, dim=(0,2,3), keepdim=True)
        g_sampleB   = (g_sampleB - g_meanB)
        g_stdB      = torch.std(g_sampleB, dim=(0,2,3), keepdim=True)
        
        d_meanB     = torch.mean(d_sampleB, dim=(0,2,3), keepdim=True)
        d_sampleB   = (d_sampleB - d_meanB)
        d_stdB      = torch.std(d_sampleB, dim=(0,2,3), keepdim=True)
        
        self.meanA  = (g_meanA + d_meanA)/2.0
        self.stdA   = (g_stdA + d_stdA)/2.0
        self.meanB  = (g_meanB + d_meanB)/2.0
        self.stdB   = (g_stdB + d_stdB)/2.0
        
        self.meanA  = self.meanA.squeeze().unsqueeze(0).unsqueeze(0).contiguous().cpu().numpy()
        self.stdA   = self.stdA.squeeze().unsqueeze(0).unsqueeze(0).contiguous().cpu().numpy()
        self.meanB  = self.meanB.squeeze().unsqueeze(0).unsqueeze(0).contiguous().cpu().numpy()
        self.stdB   = self.stdB.squeeze().unsqueeze(0).unsqueeze(0).contiguous().cpu().numpy()
        print('meanA,  stdA', self.meanA.flatten(), self.stdA.flatten())
        print('meanB,  stdB', self.meanB.flatten(), self.stdB.flatten())
        ###########################################################################
        if conf.dVar==777:
            print('dvar=777')
            self.input_image    = (self.input_image - self.meanA)
            self.input_image    = (self.input_image/self.stdA)
            self.input_imageB   = (self.input_imageB - self.meanB)
            self.input_imageB   = (self.input_imageB/self.stdB)
        else:
            self.input_image    = (self.input_image - self.meanA)
            self.input_image    = (self.input_image*conf.dVar)
            self.input_imageB   = (self.input_imageB - self.meanB)
            self.input_imageB   = (self.input_imageB*conf.dVar)
        print('normalized minmax', self.input_image.min(), self.input_image.max(), self.input_imageB.min(), self.input_imageB.max())
        ###########################################################################
        """
        
        
        self.g_inA = torch.zeros(self.bSize, 3, self.g_input_shape, self.g_input_shape).cuda()
        self.d_inA = torch.zeros(self.bSize, 3, self.d_input_shape, self.d_input_shape).cuda()
        if (conf.genMask == False) and (conf.dscMask == False):
            print('no g_inB d_inB mg_inB md_inB')
        else:
            self.g_inB = torch.zeros(self.bSize, 3, self.g_input_shape, self.g_input_shape).cuda()
            self.d_inB = torch.zeros(self.bSize, 3, self.d_input_shape, self.d_input_shape).cuda()
            self.mg_inB = torch.zeros(self.bSize, 1, 20,20).cuda() #self.g_input_shape, self.g_input_shape).cuda()
            self.md_inB = torch.zeros(self.bSize, 1, 20,20).cuda()
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx, conf):
        index = idx*self.bSize
        if (conf.genMask == True) and (conf.dscMask == True):
            for iii in range(self.bSize):
                jjj = index + iii
                # self.g_inA[iii,:,:,:], g_topA, g_leftA = self.next_crop(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                # self.d_inA[iii,:,:,:], d_topA, d_leftA = self.next_crop(for_g=False, idx=jjj) #size 26
                self.g_inB[iii,:,:,:], g_topB, g_leftB, self.mg_inB[iii,:,:,:] = self.next_crop_mask(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                self.d_inB[iii,:,:,:], d_topB, d_leftB, self.md_inB[iii,:,:,:] = self.next_crop_mask(for_g=False, idx=jjj) #size 26
            return -1, -1, self.g_inB, self.d_inB, self.mg_inB.detach(), self.md_inB.detach() # 16, 64, 26, 26
        elif (conf.genMask == False) and (conf.dscMask == False):
            for iii in range(self.bSize):
                jjj = index + iii
                self.g_inA[iii,:,:,:], g_topA, g_leftA = self.next_crop(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                self.d_inA[iii,:,:,:], d_topA, d_leftA = self.next_crop(for_g=False, idx=jjj) #size 26
                # self.g_inB[iii,:,:,:], g_topB, g_leftB, self.mg_inB[iii,:,:,:] = self.next_crop_mask(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                # self.d_inB[iii,:,:,:], d_topB, d_leftB, self.md_inB[iii,:,:,:] = self.next_crop_mask(for_g=False, idx=jjj) #size 26
            return self.g_inA, self.d_inA, -1, -1, -1, -1 # 16, 64, 26, 26
        elif (conf.genMask == True) and (conf.dscMask == False):
            for iii in range(self.bSize):
                jjj = index + iii
                self.g_inA[iii,:,:,:], g_topA, g_leftA = self.next_crop(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                self.d_inA[iii,:,:,:], d_topA, d_leftA = self.next_crop(for_g=False, idx=jjj) #size 26
                self.g_inB[iii,:,:,:], g_topB, g_leftB, self.mg_inB[iii,:,:,:] = self.next_crop_mask(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                # self.d_inB[iii,:,:,:], d_topB, d_leftB, self.md_inB[iii,:,:,:] = self.next_crop_mask(for_g=False, idx=jjj) #size 26
            return self.g_inA, self.d_inA, self.g_inB, -1, self.mg_inB.detach(), -1 # 16, 64, 26, 26
        elif (conf.genMask == False) and (conf.dscMask == True):
            for iii in range(self.bSize):
                jjj = index + iii
                self.g_inA[iii,:,:,:], g_topA, g_leftA = self.next_crop(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                # self.d_inA[iii,:,:,:], d_topA, d_leftA = self.next_crop(for_g=False, idx=jjj) #size 26
                self.g_inB[iii,:,:,:], g_topB, g_leftB, self.mg_inB[iii,:,:,:] = self.next_crop_mask(for_g=True, idx=jjj) #size 64 (for_g=True ==> select in small map: receptive field is large)
                self.d_inB[iii,:,:,:], d_topB, d_leftB, self.md_inB[iii,:,:,:] = self.next_crop_mask(for_g=False, idx=jjj) #size 26
            return self.g_inA, -1, self.g_inB, self.d_inB, self.mg_inB.detach(), self.md_inB.detach() # 16, 64, 26, 26
        else:
            print(ddd)

        #######################################
        # sumggg = self.mg_inB.sum()
        # sumddd = self.md_inB.sum()
        # summask1 = (sumddd - sumggg)/2.0 + sumggg
        # summask2 = (sumggg - sumddd)/2.0 + sumddd
        # self.mg_inB = self.mg_inB/sumggg*summask1
        # self.md_inB = self.md_inB/sumddd*summask2
        #######################################
        # conf.g_topA[iii] = g_topA
        # conf.g_leftA[iii] = g_leftA
        # conf.g_topB[iii] = g_topB
        # conf.g_leftB[iii] = g_leftB
        
    def next_crop_mask(self, for_g, idx, cropsize=None):
        if cropsize is None:
            size = self.g_input_shape if for_g else self.d_input_shape
        else:
            size = cropsize
        top, left = self.get_top_leftB(size, for_g, idx)
        crop_im = self.input_imageB[top:top + size, left:left + size, :]
        if for_g is True:
            crop_mk = self.maskB[top:top + size, left:left + size, :]
        else:
            crop_mk = self.maskBd[top:top + size, left:left + size, :]
        mask = torch.tensor(crop_mk).permute(2,0,1).unsqueeze(0)
        mask = F.max_pool2d(mask[:,:,6:-6,6:-6],2,stride=2)[:,0,3:-3,3:-3] if for_g else mask[:,0,3:-3,3:-3]
        
        if not for_g:  # Add noise to the image for d
            crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im), top, left, mask.cuda()
        
    def next_crop(self, for_g, idx, cropsize=None):
        if cropsize is None:
            size = self.g_input_shape if for_g else self.d_input_shape
        else:
            size = cropsize
        top, left = self.get_top_left(size, for_g, idx)
        crop_im = self.input_image[top:top + size, left:left + size, :]
            
        if not for_g:  # Add noise to the image for d
            crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im), top, left

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        if size%2 == 1: # size=7
            top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        else:           # size=64, 26 and 26
            if idx%4==0:
                top, left = min(max(0, row -(size//2 - 1)), self.in_rows - size), min(max(0, col -(size//2 - 1)), self.in_cols - size)
            elif idx%4==1:
                top, left = min(max(0, row -(size//2)), self.in_rows - size), min(max(0, col -(size//2 - 1)), self.in_cols - size)
            elif idx%4==2:
                top, left = min(max(0, row -(size//2 - 1)), self.in_rows - size), min(max(0, col -(size//2)), self.in_cols - size)
            elif idx%4==3:
                top, left = min(max(0, row -(size//2)), self.in_rows - size), min(max(0, col -(size//2)), self.in_cols - size)
            else:
                print(ddd)
        return top, left
        # Choose even indices (to avoid misalignment with the loss map for_g)
        # return top - top % 2, left - left % 2
        

    def get_top_leftB(self, size, for_g, idx):
        center = self.crop_indices_for_gB[idx] if for_g else self.crop_indices_for_dB[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        if size%2 == 1: # size=7
            top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        else:           # size=64, 26 and 26
            if idx%4==0:
                top, left = min(max(0, row -(size//2 - 1)), self.in_rows - size), min(max(0, col -(size//2 - 1)), self.in_cols - size)
            elif idx%4==1:
                top, left = min(max(0, row -(size//2)), self.in_rows - size), min(max(0, col -(size//2 - 1)), self.in_cols - size)
            elif idx%4==2:
                top, left = min(max(0, row -(size//2 - 1)), self.in_rows - size), min(max(0, col -(size//2)), self.in_cols - size)
            elif idx%4==3:
                top, left = min(max(0, row -(size//2)), self.in_rows - size), min(max(0, col -(size//2)), self.in_cols - size)
            else:
                print(ddd)
        return top, left
        


    def make_list_of_crop_indices(self, conf):
        iterations = conf.max_iters*conf.bSize
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor, conf=conf)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d#, crop_indices_for_fake1, crop_indices_for_fake2


    def make_list_of_crop_indicesB(self, conf):
        iterations = conf.max_iters*conf.bSize
        prob_map_big, prob_map_sml = self.create_prob_mapsB(scale_factor=conf.scale_factor, conf=conf)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d#, crop_indices_for_fake1, crop_indices_for_fake2
        
        
    def create_prob_maps(self, scale_factor, conf): # scale_factor = 0.5
        loss_map_big = create_gradient_map(self.input_image)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        prob_map_big, prob_nof_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml, prob_nof_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        
        #####################################################
        # prob_nof_big = prob_nof_big/prob_nof_big.max()
        # prob_nof_sml = prob_nof_sml/prob_nof_sml.max()
        
        # max_val = 255 if prob_nof_big.dtype == 'uint8' else 1.
        # if not conf.X4:
            # plt.imsave(os.path.join('%s/prob_mapx2_sml_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_sml, vmin=0, vmax=max_val, dpi=1)
            # plt.imsave(os.path.join('%s/prob_mapx2_big_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_big, vmin=0, vmax=max_val, dpi=1)
        # else:
            # plt.imsave(os.path.join('%s/prob_mapx4_sml_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_sml, vmin=0, vmax=max_val, dpi=1)
            # plt.imsave(os.path.join('%s/prob_mapx4_big_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_big, vmin=0, vmax=max_val, dpi=1)
        #####################################################

        return prob_map_big, prob_map_sml



    def create_prob_mapsB(self, scale_factor, conf): # scale_factor = 0.5
        # loss_map_big = create_gradient_map(self.input_imageB)
        # loss_map_sml = create_gradient_map(imresize(im=self.input_imageB, scale_factor=scale_factor, kernel='cubic'))
        # prob_map_big, prob_nof_big = create_probability_map(loss_map_big, self.d_input_shape)
        # prob_map_sml, prob_nof_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        
        # prob_map_big = prob_map_big * self.maskB[:,:,0].flatten()
        # prob_map_sml = prob_map_sml * self.maskB[:,:,0].flatten()
        
        # prob_nof_big = prob_nof_big * self.maskB[:,:,0]
        # prob_nof_sml = prob_nof_sml * self.maskB[:,:,0]
        prob_map_big = self.maskBd[:,:,0].flatten()
        prob_map_sml = self.maskB[:,:,0].flatten()
        
        # prob_nof_big = self.maskBd[:,:,0]
        # prob_nof_sml = self.maskB[:,:,0]
        if prob_map_big.sum() != 0:
            prob_map_big = prob_map_big / prob_map_big.sum() 
        else:
            print(ddd)
        
        if prob_map_sml.sum() != 0:
            prob_map_sml = prob_map_sml / prob_map_sml.sum() 
        else:
            print(ddd)
        #####################################################
        # prob_nof_big = prob_nof_big/prob_nof_big.max()
        # prob_nof_sml = prob_nof_sml/prob_nof_sml.max()
        
        # max_val = 255 if prob_nof_big.dtype == 'uint8' else 1.
        # if not conf.X4:
            # plt.imsave(os.path.join('%s/probB_mapx2_sml_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_sml, vmin=0, vmax=max_val, dpi=1)
            # plt.imsave(os.path.join('%s/probB_mapx2_big_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_big, vmin=0, vmax=max_val, dpi=1)
        # else:
            # plt.imsave(os.path.join('%s/probB_mapx4_sml_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_sml, vmin=0, vmax=max_val, dpi=1)
            # plt.imsave(os.path.join('%s/probB_mapx4_big_%s.png' % (conf.output_dir_path, conf.img_name)), prob_nof_big, vmin=0, vmax=max_val, dpi=1)
        #####################################################

        return prob_map_big, prob_map_sml


    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = 4 #int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image
