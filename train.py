from data_loader import HDRDataset
from options import Options
from torch.utils.data import Dataset, DataLoader
from model import FHDR
import os, time, torch, torch.nn as nn
import numpy as np
from util import mu_tonemap, save_ldr_image, save_hdr_image, save_checkpoint, update_lr
from vgg import VGGLoss
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.data.normal_(0.0, 0.0)

# initialise training/testing params
opt = Options().parse()

# loading data
data = HDRDataset(mode='train', opt=opt)
dataset = DataLoader(data, batch_size=opt.batch_size)

print("Training samples: ", len(data))

# loading the model
model = FHDR(iteration_count = opt.iter)

str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set gpu ids
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

# initialising losses
l1 = torch.nn.L1Loss()
perceptual_loss = VGGLoss()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

if torch.cuda.device_count() > 0:
    assert(torch.cuda.is_available())   
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

if not os.path.exists('./checkpoints'):
    print('Making checkpoints directory')
    os.makedirs('./checkpoints')

if not os.path.exists('./training_results'):
    print('Making training_results directory')
    os.makedirs('./training_results')

if opt.continue_train:
    try:
        start_epoch = np.loadtxt('./checkpoints/state.txt', dtype=int)
        model.load_state_dict(torch.load(opt.ckpt_path))
        print('Resuming from epoch ', start_epoch)

    except:
        start_epoch = 1
        model.apply(weights_init)
        print('Checkpoint not found! Training from scratch.')
else:    
    start_epoch = 1
    model.apply(weights_init)

if opt.print_model:
    print(model)

for epoch in range(start_epoch, opt.epochs + 1):
    
    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)

    epoch_start = time.time()
    running_loss = 0

    print("Epoch: ", epoch)

    for batch, data in enumerate(tqdm(dataset, desc='Batch %')):

        optimizer.zero_grad()
        
        input = data['ldr_image'].data.cuda()
        ground_truth = data['hdr_image'].data.cuda()

        # forward pass
        output = model(input)

        l1_loss = 0
        vgg_loss = 0
        
        # tonemapping ground truth
        mu_tonemap_gt = mu_tonemap(ground_truth)

        # computing loss for n outputs (from n-iterations)
        for image in output:
            l1_loss += l1(mu_tonemap(image), mu_tonemap_gt)
            vgg_loss += perceptual_loss(mu_tonemap(image), mu_tonemap_gt)

        l1_loss /= len(output)
        vgg_loss /= len(output)
        
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        loss = l1_loss + (vgg_loss * 10)

        output = output[-1]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch + 1) % opt.log_after == 0:    # logging batch count and loss value 
            print("Epoch: {} ; Batch: {} ; Training loss: {}".format(epoch, batch + 1, running_loss/opt.log_after))
            running_loss = 0
            
        if (batch + 1) % opt.save_results_after == 0:    # save image results
            save_ldr_image(img_tensor=input, batch=0, path='./training_results/ldr_e_{}_b_{}.jpg'.format(epoch, batch+1))
            save_hdr_image(img_tensor=output, batch=0, path='./training_results/generated_hdr_e_{}_b_{}.hdr'.format(epoch, batch+1))
            save_hdr_image(img_tensor=ground_truth, batch=0, path='./training_results/gt_hdr_e_{}_b_{}.hdr'.format(epoch, batch+1))

    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)//60
    
    print('End of epoch {}. Time taken: {} minutes.'.format(epoch, int(time_taken)))

    if (epoch % opt.save_ckpt_after == 0):
        save_checkpoint(epoch, model)

print('Training complete!')