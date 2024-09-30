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

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64, normalize=True).to("cuda")
fid_object = FrechetInceptionDistance(feature=64, normalize=True).to("cuda")

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
fids = []
ssims_object = []
psnrs_object = []
lpipss_object = []
fids_object = []

# import subprocess
# res = subprocess.run(["python", "-m", "pytorch_fid", img_dir, gt_dir])
# print(res)
for i, img_name in tqdm(enumerate(natsorted(os.listdir(img_dir)))):
    img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
    gt = Image.open(os.path.join(gt_dir, gt_list[i])).convert("RGB")
    object_mask = Image.open(os.path.join(object_mask_dir, gt_list[i]))
    
    img_tensor = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()  
    object_mask_tensor = tf.to_tensor(object_mask).unsqueeze(0)[:, :3, :, :].cuda()
    object_mask_tensor[object_mask_tensor > 0.5] = 1
    
    # breakpoint()
    # calculate metrics
    ssims.append(ssim(img_tensor, gt_tensor))
    psnrs.append(psnr(img_tensor, gt_tensor))
    lpipss.append(lpips(img_tensor, gt_tensor, net_type='vgg'))
    
    fid.update(img_tensor, real=False)  
    fid.update(gt_tensor, real=True)
    
    # calculate metrics for object only
    img_tensor_object = img_tensor * object_mask_tensor
    gt_tensor_object = gt_tensor * object_mask_tensor
    ssims_object.append(ssim(img_tensor_object, gt_tensor_object))
    psnrs_object.append(psnr(img_tensor_object, gt_tensor_object)) 
    lpipss_object.append(lpips(img_tensor_object, gt_tensor_object, net_type='vgg'))
    fid_object.update(img_tensor_object, real=False)
    fid_object.update(gt_tensor_object, real=True)
    

# res = subprocess.run(["python", "-m", "pytorch_fid", img_dir, gt_dir], capture_output=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

# print("==========Full Image==========")    
# print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".2"))
# print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".2"))
# print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".2"))
# print("==========Object==========")
# print("  SSIM : {:>12.7f}".format(torch.tensor(ssims_object).mean(), ".2"))
# print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs_object).mean(), ".2"))
# print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss_object).mean(), ".2"))

print("========== ==========\n")    
print(f"{torch.tensor(ssims).mean():.3f} {torch.tensor(ssims_object).mean():.3f}")
print(f"{torch.tensor(psnrs).mean():.3f} {torch.tensor(psnrs_object).mean():.3f}")
print(f"{torch.tensor(lpipss).mean():.3f} ({torch.tensor(lpipss_object).mean():.3f})")
print(f"{fid.compute():.3f} {fid_object.compute():.3f}") 





