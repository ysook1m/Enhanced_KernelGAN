# Enhanced_KernelGAN
Unsupervised Blur Kernel Estimation and Correction for Blind Super-Resolution

paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9762718

appendix: https://ieeexplore.ieee.org/ielx7/6287639/9668973/9762718/supp1-3170053.pdf?arnumber=9762718

# Dataset
download DIV2KRK from https://github.com/sefibk/KernelGAN

# Usage
python train.py --input-dir input_image_folder --output-dir output_kernel_folder --real --dVar 777.0 --vCenter 0.8 --wSTD 0.01

# Code Reference
https://github.com/sefibk/KernelGAN
