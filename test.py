########################################
import os
import argparse
import logging
from collections import OrderedDict
import test_util as util

logger = logging.getLogger('base')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='../data/DIV2KRK/gt')
parser.add_argument('-d1','--dir1', type=str, default='./results_real/img_x2_raw')

opt = parser.parse_args()

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []

# crawl directories
files = os.listdir(opt.dir0)
print(opt.dir1)
# print(files)

for file in files:
    # Load images
    gt_img = util.load_image(os.path.join(opt.dir0,file)) # RGB image from [-1,1]
    file2 = 'ZSSR_' + file
    # file2 = file
    
    sr_img = util.load_image(os.path.join(opt.dir1,file2))


    # calculate PSNR and SSIM
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
    ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)

    if gt_img.shape[2] == 3:  # RGB image
        sr_img_y = util.bgr2ycbcr(sr_img, only_y=True)
        gt_img_y = util.bgr2ycbcr(gt_img, only_y=True)

        cropped_sr_img_y = sr_img_y
        cropped_gt_img_y = gt_img_y

        psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
        ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
        test_results['psnr_y'].append(psnr_y)
        test_results['ssim_y'].append(ssim_y)
        
        
        logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            format(file, psnr, ssim, psnr_y, ssim_y))
        print('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            format(file, psnr, ssim, psnr_y, ssim_y))
    else:
        logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(file, psnr, ssim))
        print('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(file, psnr, ssim))

# Average PSNR/SSIM results
ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
logger.info('----Average PSNR/SSIM results ----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        ave_psnr, ave_ssim))
print('----Average PSNR/SSIM results ----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        ave_psnr, ave_ssim))
if test_results['psnr_y'] and test_results['ssim_y']:
    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
    logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
        format(ave_psnr_y, ave_ssim_y))
    print('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
        format(ave_psnr_y, ave_ssim_y))