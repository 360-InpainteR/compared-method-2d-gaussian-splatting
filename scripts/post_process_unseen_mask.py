import argparse
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import cv2


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", "-s", type=str, required=True)
    argparser.add_argument("--dataset", "-d", type=str, required=True, choices=["360", "our", "bear"], default="bear")
    argparser.add_argument("--exp", "-e", type=str, default="exp1")
    argparser.add_argument("--dilate_iter", "-i", type=int, default=3)
    argparser.add_argument("--kernel_size", "-k", type=int, default=5)
   
    args = argparser.parse_args()
    
    if args.dataset == '360':
        dataset_name = '360v2_with_masks'
    elif args.dataset == 'our':
        dataset_name = 'our_dataset'
    else:
        dataset_name = ""
        
    
    unseen_mask_dir = f'output/{dataset_name}/{args.scene}/{args.exp}/train/ours_30000_object_removal/unseen_mask/'
    name_dir = f'data/{dataset_name}/{args.scene}/images/'
    output_dir = f'data/{dataset_name}/{args.scene}/unseen_mask/'
    os.makedirs(output_dir, exist_ok=True)
    
    name_list = natsorted(os.listdir(name_dir))
    
    dilate_iter = args.dilate_iter
    kernel_size = args.kernel_size
    
    for i, file in enumerate(natsorted(os.listdir(unseen_mask_dir))):
        file_path = os.path.join(unseen_mask_dir, file)
        if os.path.isfile(file_path):
            unseen_mask = np.array(Image.open(os.path.join(unseen_mask_dir, file)))
            unseen_mask = (unseen_mask > 0).astype(np.uint8) * 255
            
            # process the mask
            try: # handle the case where the mask is empty
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                cleaned_mask = cv2.morphologyEx(unseen_mask, cv2.MORPH_OPEN, kernel)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
                largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                cleaned_mask = np.zeros_like(labels)
                cleaned_mask[labels == largest_component] = 255
                cleaned_mask = cleaned_mask.astype(np.uint8)
                cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=dilate_iter)
            except:
                cleaned_mask = np.zeros_like(unseen_mask)
            
            # save the mask
            Image.fromarray(cleaned_mask).save(os.path.join(output_dir, os.path.splitext(name_list[i])[0] + '.png'))
    
    
    render_dir = f'output/{dataset_name}/{args.scene}/{args.exp}/train/ours_30000_object_removal/renders/'
    
    os.makedirs(f"tmp/{dataset_name}/", exist_ok=True)
    os.system(f"python scripts/visualize_mask.py --img_dir {render_dir} --mask_dir {output_dir} --output_file_name tmp/{dataset_name}/{args.scene}.mp4")