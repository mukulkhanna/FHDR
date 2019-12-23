from data_loader import HDRDataset
from options import Options
from torch.utils.data import Dataset, DataLoader
from model import FHDR
import os, time, torch, torch.nn as nn
import numpy as np
from util import mu_tonemap, save_ldr_image, save_hdr_image
from vgg import VGGLoss
from skimage.measure import compare_ssim
from tqdm import tqdm

opt = Options().parse()

hdr_dataset = HDRDataset(mode='test', opt=opt)
dataset = DataLoader(hdr_dataset, batch_size=opt.batch_size)

print("Testing samples: ", len(hdr_dataset))

model = FHDR(iteration_count = opt.iter)

str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

if torch.cuda.device_count() > 0:
    assert(torch.cuda.is_available())   
    model.cuda()

if not os.path.exists('./test_results'):
    print('Making test_results directory')
    os.makedirs('./test_results')

model.load_state_dict(torch.load(opt.ckpt_path))
    
avg_psnr = 0
avg_ssim = 0

print('Starting evaluation. Results will be saved in \'/test_results\' directory')

with torch.no_grad():    
    
    for batch, data in enumerate(tqdm(dataset, desc="Testing %")):

        optimizer.zero_grad()
        
        input = data['ldr_image'].data.cuda()
        ground_truth = data['hdr_image'].data.cuda()

        output = model(input)
        
        mu_tonemap_gt = mu_tonemap(ground_truth)

        output = output[-1]
        
        for batch_ind in range(len(output.data)):

            save_ldr_image(img_tensor=input, batch=batch_ind, path='./test_results/ldr_b_{}_{}.png'.format(batch, batch_ind))
            save_hdr_image(img_tensor=output, batch=batch_ind, path='./test_results/generated_hdr_b_{}_{}.hdr'.format(batch, batch_ind))
            save_hdr_image(img_tensor=ground_truth, batch=batch_ind, path='./test_results/gt_hdr_b_{}_{}.hdr'.format(batch, batch_ind))
            
            if opt.log_scores:
                mse = mse_loss(mu_tonemap(output.data[batch_ind]), mu_tonemap_gt.data[batch_ind])
                psnr = (10 * np.log10(1 / mse.item()))

                avg_psnr +=  psnr

                generated = (np.transpose(output.data[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                real = (np.transpose(ground_truth.data[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                ssim = compare_ssim(generated, real, multichannel = True)
                avg_ssim += ssim

if opt.log_scores:
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(hdr_dataset)))
    print("Avg SSIM -> " + str(avg_ssim/len(hdr_dataset)))

print('Evaluation completed.')
