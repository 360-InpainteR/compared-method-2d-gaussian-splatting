import os
import argparse
import numpy as np
from PIL import Image
from natsort import natsorted
import cv2

# python scripts/gen_unseen_mask.py -i output/bear_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000_object_removal/seen_mask/ -o output/bear_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000/object_mask/ -s 127
# python scripts/gen_unseen_mask.py -i output/kitchen_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000_object_removal/seen_mask/ -o output/kitchen_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000/object_mask/ -s 250
# python scripts/gen_unseen_mask.py --scene room -o output/360_v2_with_masks/incomplete_isMasked_3dim_detach_nomeanloss/room/train/ours_30000/object_mask/ -i output/360_v2_with_masks/incomplete_isMasked_3dim_detach_nomeanloss/room/train/ours_30000_object_removal/seen_mask/ --seen_thr 127 -k 3 --open_iter 3 --dialate_iter 5

def check_thr(value):
    if type(value) is int:
        raise argparse.ArgumentTypeError(f"Threshold {value} is not an integer")
    ivalue = int(value)
    if ivalue < 0 or ivalue > 255:
        raise argparse.ArgumentTypeError(f"Threshold {value} is out of range (0-255)")
    return ivalue

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", type=str, required=True)
    argparser.add_argument("--object_mask_dir", "-o", type=str, required=True)
    argparser.add_argument("--isseen_mask_dir", "-i", type=str, required=True)
    # seen_thrr region [0, 255]
    argparser.add_argument("--seen_thr", "-s", type=check_thr, required=True, default=127)
    argparser.add_argument("--kernel_size", "-k", type=int, required=False, default=5)
    argparser.add_argument("--open_iter", type=int, required=False, default=5)
    argparser.add_argument("--dialate_iter", type=int, required=False, default=10)
    
    args = argparser.parse_args()
    
    output_root = f'data/360_v2_with_masks/{args.scene}/inpaint_2d_unseen_mask/'
    os.makedirs(output_root, exist_ok=True)
    
    name_dir = f'data/360_v2_with_masks/{args.scene}/images/'
    name_list = natsorted(os.listdir(name_dir))
    # read all files (not dir) under object_mask_dir
    object_mask_dir = args.object_mask_dir
    isseen_mask_dir = args.isseen_mask_dir
    
    # adjust mask
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    open_iter = args.open_iter
    dialate_iter = args.dialate_iter
    
    
    
    index = 0
    for filename in os.listdir(object_mask_dir):
        if os.path.isfile(os.path.join(object_mask_dir, filename)):
            # read png file
            object_mask = np.array(Image.open(os.path.join(object_mask_dir, filename)))
            
            object_mask = (object_mask > 0).astype(bool)
            
            isseen_mask = np.array(Image.open(os.path.join(isseen_mask_dir, filename)))
            isseen_mask = (isseen_mask > args.seen_thr).astype(bool)
            
            unseen_mask = np.logical_and(object_mask, np.logical_not(isseen_mask))
            unseen_mask = unseen_mask.astype(np.uint8) * 255
            
            
            # get root dir of isseen_mask_dir
            unseen_mask_dir = os.path.join(os.path.dirname(os.path.dirname(isseen_mask_dir)), "unseen_mask")
            os.makedirs(unseen_mask_dir, exist_ok=True)           
            Image.fromarray(unseen_mask).save(os.path.join(unseen_mask_dir, filename))
            
            # save to dat/scene/inpaint_2d_unseen_mask
                # do dialation
            
            cleaned_mask = cv2.morphologyEx(unseen_mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)

                # Filter components based on their area (keeping the largest component)
            try:
                largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                cleaned_mask = np.zeros_like(labels)
                cleaned_mask[labels == largest_component] = 255
                cleaned_mask = cleaned_mask.astype(np.uint8)
                # dialation
            except:
                pass
            cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=dialate_iter)
            
            
            
            Image.fromarray(cleaned_mask).save(os.path.join(output_root, os.path.splitext(name_list[index])[0] + '.png'))
            # Image.fromarray(cleaned_mask).save(os.path.join(output_root, filename))
            index += 1
            
    