import argparse
import os
import sys

sys.path.append(".")

import torch
import torchvision.transforms.functional as tf
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim

argparser = argparse.ArgumentParser()
argparser.add_argument("--scene", "-s", type=str, required=True)
argparser.add_argument("--exp", "-e", type=str, default="exp1")

args = argparser.parse_args()
dataset_name = "our_dataset"
scene_name = args.scene
exp_name = args.exp

gt_dir = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/our_dataset/{scene_name}/test_images/"
gt_list = natsorted(os.listdir(gt_dir))
img_dir = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/our_dataset/{scene_name}/{exp_name}/test/ours_10000_object_inpaint/renders/"
object_mask_dir = f"/project/gs-inpainting/data/our_dataset/{scene_name}/test_object_masks/"

scenes = ["box", "cone", "cookie", "dutpan", "lawn", "plant", "sunflower"]

ssims = []
psnrs = []
lpipss = []
ssims_object = []
psnrs_object = []
lpipss_object = []

import subprocess
res = subprocess.run(["python", "-m", "pytorch_fid", img_dir, gt_dir])
print(res)

for i, img_name in tqdm(enumerate(natsorted(os.listdir(img_dir)))):
    img = Image.open(os.path.join(img_dir, img_name))
    gt = Image.open(os.path.join(gt_dir, gt_list[i]))
    object_mask = Image.open(os.path.join(object_mask_dir, gt_list[i]))
    
    img_tensor = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()  
    object_mask_tensor = tf.to_tensor(object_mask).unsqueeze(0)[:, :3, :, :].cuda()
    object_mask_tensor[object_mask_tensor > 0.5] = 1
    
    
    # calculate metrics
    ssims.append(ssim(img_tensor, gt_tensor))
    psnrs.append(psnr(img_tensor, gt_tensor))
    lpipss.append(lpips(img_tensor, gt_tensor, net_type='vgg'))
    
    # calculate metrics for object only
    img_tensor_object = img_tensor * object_mask_tensor
    gt_tensor_object = gt_tensor * object_mask_tensor
    ssims_object.append(ssim(img_tensor_object, gt_tensor_object))
    psnrs_object.append(psnr(img_tensor_object, gt_tensor_object)) 
    lpipss_object.append(lpips(img_tensor_object, gt_tensor_object, net_type='vgg'))
    

os.system(f"python -m pytorch_fid {img_dir} {gt_dir}")



print("==========Full Image==========")    
print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".2"))
print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".2"))
print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".2"))
print("==========Object==========")
print("  SSIM : {:>12.7f}".format(torch.tensor(ssims_object).mean(), ".2"))
print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs_object).mean(), ".2"))
print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss_object).mean(), ".2"))

print("========== ==========\n")    
print(f"{torch.tensor(ssims).mean():.2f} ({torch.tensor(ssims_object).mean():.2f}) {torch.tensor(lpipss).mean():.2f} ({torch.tensor(psnrs_object).mean():.2f}) {torch.tensor(lpipss).mean():.2f} ({torch.tensor(lpipss_object).mean():.2f})")
    





