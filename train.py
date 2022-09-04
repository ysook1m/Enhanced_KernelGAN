import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import natsort 
from util import im2tensor

genMask    = False #True False
dscMask    = False #True False

gantype    = 'GAN' #GAN LSL1 LSL2
rankDisc   = True #True False
wSTD       = 0.01

# gantype    = 'LSL1' #GAN LSL1 LSL2
# rankDisc   = False #True False
# wSTD       = 1.0

def train(conf):
    if not conf.onlySR:
        gan = KernelGAN(conf)
        learner = Learner()
        data = DataGenerator(conf, gan)
        gan.dip_input(data.in_dip)
        gan.initialize_kernel()
        # for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60, mininterval=60.0):
            conf.iter = iteration
            [g_inA, d_inA, g_inB, d_inB, mg_inB, md_inB] = data.__getitem__(iteration, conf)
            gan.train(g_inA, d_inA, g_inB, d_inB, mg_inB, md_inB, conf)
            learner.update(iteration, gan)
            
        gan.log_save()
        gan.finish()
        
        # conf.iter = 77777
        # dadata = im2tensor(data.input_image)
        # gan.train(dadata, None, None, None, None, None, conf)
        
    else:
        gan = KernelGAN(conf)
        learner = Learner()
        data = DataGenerator(conf, gan)
        gan.do_ZSSR()


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results_real', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=0.5, help='ZSSR uses this to partially de-noise images')
    prog.add_argument('--onlySR', action='store_true', help='only ZSSRs')
    # prog.add_argument('--dPosEnc', action='store_true', help='positional encoding for discriminator')
    prog.add_argument('--wSTD', type=float, default=777.0, help='# of batch')
    prog.add_argument('--dVar', type=float, default=1.0, help='# of batch')
    
    prog.add_argument('--rankDisc', action='store_true', help='rankDisc')
    prog.add_argument('--genMask', action='store_true', help='genMask')
    prog.add_argument('--dscMask', action='store_true', help='dscMask')
    prog.add_argument('--gantype', type=str, default='GAN', help='gantype')
    prog.add_argument('--g_thc', type=int, default=5, help='thick')
    prog.add_argument('--d_thc', type=int, default=5, help='thick')
    prog.add_argument('--vCenter', type=float, default=0.9, help='# of batch')

    args = prog.parse_args()
    args.wSTD = wSTD
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in natsort.natsorted(os.listdir(os.path.abspath(args.input_dir))):
        conf = Config().parse(create_params(filename, args))
        conf.onlySR     = args.onlySR
        
        conf.dVar       = args.dVar
        conf.wSTD       = args.wSTD
        conf.rankDisc   = rankDisc
        conf.genMask    = genMask
        conf.dscMask    = dscMask
        conf.gantype    = gantype
        conf.g_thc      = args.g_thc
        conf.d_thc      = args.d_thc
        conf.vCenter      = args.vCenter


        ###############################################
        fname = os.path.join(conf.output_dir_path, 'real_x2_777/nonc_%s_kernel_x2.mat' % (filename.split('.')[0]))
        print(' :', os.path.isfile(fname), os.getcwd(), filename)
        if os.path.isfile(fname):
            continue
        
        fname = os.path.join(conf.output_dir_path, './nonc_%s_kernel_x2' % (filename.split('.')[0]))
        if os.path.isfile(fname):
            continue
        f = open(fname, 'w')
        f.close()
        ###############################################
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    print('#####################################################', args.input_dir.split('/')[-2])
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir)+'_'+args.input_dir.split('/')[-2] + '_w' + str(args.wSTD) + '_dvar' + str(args.dVar) + '_g' + str(args.g_thc) + '_d' + str(args.d_thc) + '_v' + str(args.vCenter),
              '--noise_scale', str(args.noise_scale)]
              # '--output_dir_path', os.path.abspath(args.output_dir)+'_'+args.input_dir.split('/')[-2] + '_' + str(args.wSTD) + '_rank' + str(args.rankDisc) + '_gM' + str(args.genMask) + '_dM' + str(args.dscMask) + '_' + str(args.gantype),
    if args.X4:
        params.append('--X4')
    if args.real:
        params.append('--real_image')
    if args.onlySR:
        params.append('--onlySR')
    # if args.dPosEnc:
        # params.append('--dPosEnc')
    return params


if __name__ == '__main__':
    main()
