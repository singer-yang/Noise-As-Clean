import os
import logging
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import cv2 as cv

from torchvision.utils import save_image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.utils import set_seed
from utils.single_img import NAC_singleImg, generate_noisy
from utils.network import DnCNN
from utils import utils_logger
from utils.utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)


# --------------------------- parameters --------------------------- 
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--noise1", type=float, default=15, help="gaussian noise 1 level")
parser.add_argument("--noise2", type=float, default=15, help="gaussian noise 2 level")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--epochs", type=int, default=800, help="epochs to run")
parser.add_argument("--eval_every", type=int, default=100, help="evaluate the model every")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
parser.add_argument("--num_of_layers", type=int, default=7, help="number of DnCNN layers")
args = parser.parse_args()


# --------------------------- logging --------------------------------------
result_dir = f'./results/single_img/sigma{args.noise1}-{args.noise2}-' + datetime.now().strftime("%m%d-%H%M")
os.makedirs(result_dir, exist_ok =True)

logger_name = 'train'
utils_logger.logger_info(logger_name, f'{result_dir}/{logger_name}.log')
logger = logging.getLogger(logger_name)
logger.info(args)
logger.info(f'===> save log file into {result_dir}. \n')
writer = SummaryWriter(result_dir)


# --------------------------- load data -------------------------------------
imgx0, imgy0 = generate_noisy(args.noise1, './datasets/Set14/lenna.png')
cv.imwrite(f'{result_dir}/clean.png', cv.cvtColor(imgx0 * 255., cv.COLOR_RGB2BGR))
cv.imwrite(f'{result_dir}/noisy.png', cv.cvtColor(imgy0 * 255., cv.COLOR_RGB2BGR))
logger.info(f'===> noisy image generated and saved. \n')

nac_dataset = NAC_singleImg(imgy0, args.noise2, mode='train')
nac_dataloader = DataLoader(nac_dataset, batch_size=4)


# --------------------------- train the model -------------------------------
model = DnCNN(channels=3, num_of_layers=args.num_of_layers)
model = nn.DataParallel(model)
model = model.cuda()

lr = args.lr
criterion = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = 4 * args.eval_every, gamma=0.1)

logger.info('===> start training!\n')
psnr_max = 0

for epoch in range(args.epochs):
    # train
    for imgy, imgz in nac_dataloader:
        imgz = imgz.cuda()
        imgy = imgy.cuda()

        model.train()
        optim.zero_grad()
        imgy_pred = model(imgz)
        loss = criterion(imgy_pred, imgy) / (imgy.size()[0]*2)

        loss.backward()
        optim.step()
        scheduler.step()
        writer.add_scalar('train_loss', loss, epoch)

    # eval
    if (epoch+1) % args.eval_every == 0:
        model.eval()
        with torch.no_grad():
            clean0 = torch.from_numpy(imgx0)
            noisy0 = torch.from_numpy(imgy0).permute(2, 0, 1).unsqueeze(0).cuda()

            img_pred0 = model(noisy0)
            save_image(img_pred0, f'{result_dir}/pred{epoch+1}.png')
            
            # compute psnr and update model
            psnr = compare_psnr(clean0.numpy(), img_pred0[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
            ssim = compare_ssim(clean0.numpy(), img_pred0[0, :, :, :].cpu().numpy().transpose(1, 2, 0), multichannel=True)
            logger.info(f"epoch{epoch + 1}, learning rate is {scheduler.get_last_lr()}, training loss is {loss}, psnr is {psnr}, ssim is {ssim}.")
            writer.add_scalar('val_psnr', psnr, epoch)

logger.info(f'training finished.')