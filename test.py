import os
import time

import numpy as np
import torch
import torch.nn as nn
from skimage.measure import compare_ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import HDRDataset
from model import FHDR
from options import Options
from util import make_required_directories, mu_tonemap, save_hdr_image, save_ldr_image
from vgg import VGGLoss

# initialise options
opt = Options().parse()

# ======================================
# loading data
# ======================================

dataset = HDRDataset(mode="test", opt=opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

print("Testing samples: ", len(dataset))

# ========================================
# model initialising, loading & gpu configuration
# ========================================

model = FHDR(iteration_count=opt.iter)

str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set gpu device
if len(opt.gpu_ids) > 0:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= len(opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    model.cuda()

mse_loss = nn.MSELoss()

# loading checkpoint for evaluation
model.load_state_dict(torch.load(opt.ckpt_path))

make_required_directories(mode="test")

avg_psnr = 0
avg_ssim = 0

print("Starting evaluation. Results will be saved in '/test_results' directory")

with torch.no_grad():

    for batch, data in enumerate(tqdm(data_loader, desc="Testing %")):

        input = data["ldr_image"].data.cuda()
        ground_truth = data["hdr_image"].data.cuda()

        output = model(input)

        # tonemapping ground truth image for PSNR-μ calculation
        mu_tonemap_gt = mu_tonemap(ground_truth)

        output = output[-1]

        for batch_ind in range(len(output.data)):

            # saving results
            save_ldr_image(
                img_tensor=input,
                batch=batch_ind,
                path="./test_results/ldr_b_{}_{}.png".format(batch, batch_ind),
            )
            save_hdr_image(
                img_tensor=output,
                batch=batch_ind,
                path="./test_results/generated_hdr_b_{}_{}.hdr".format(
                    batch, batch_ind
                ),
            )
            save_hdr_image(
                img_tensor=ground_truth,
                batch=batch_ind,
                path="./test_results/gt_hdr_b_{}_{}.hdr".format(batch, batch_ind),
            )

            if opt.log_scores:
                # calculating PSNR score
                mse = mse_loss(
                    mu_tonemap(output.data[batch_ind]), mu_tonemap_gt.data[batch_ind]
                )
                psnr = 10 * np.log10(1 / mse.item())

                avg_psnr += psnr

                generated = (
                    np.transpose(output.data[batch_ind].cpu().numpy(), (1, 2, 0)) + 1
                ) / 2.0
                real = (
                    np.transpose(ground_truth.data[batch_ind].cpu().numpy(), (1, 2, 0))
                    + 1
                ) / 2.0

                # calculating SSIM score
                ssim = compare_ssim(generated, real, multichannel=True)
                avg_ssim += ssim

if opt.log_scores:
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    print("Avg SSIM -> " + str(avg_ssim / len(dataset)))

print("Evaluation completed.")
