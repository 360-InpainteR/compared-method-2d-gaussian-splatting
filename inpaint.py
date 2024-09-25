#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from os import makedirs
from random import randint

import cv2
import lpips
import numpy as np
import open3d as o3d
import torch
import torchvision
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.loss_utils import l1_loss, ssim, masked_l1_loss
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh, to_cam_open3d
from utils.render_utils import create_videos, generate_path
from utils.depth_utils import estimate_depth, align_depth_poisson, align_depth_gradient_descent, align_depth_least_square, estimate_depth_depth_anything_v2
from utils.warping_utils import unproject, voxel_downsample
from utils.general_utils import colormap, vis_depth
from torch.utils.tensorboard import SummaryWriter



def finetune_inpaint(scene, opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, voxel_down_size, finetune_iteration):
    
    tb_writer = SummaryWriter(args.model_path+'/')
    # get reference view, current set 1st view as reference view. Then, do depth alignment
    
    with torch.no_grad():
        reference_view = views[0]
        render_pkg = render(reference_view, gaussians, pipe, background)
        rgb, depth = render_pkg["render"], render_pkg["surf_depth"]
        inpaint_rgb = reference_view.original_image
        unseen_mask = torch.tensor(reference_view.original_image_mask).cuda()
        
        torchvision.utils.save_image(rgb, f"tmp/{reference_view.image_name}_incomplete.png")
        torchvision.utils.save_image(inpaint_rgb, f"tmp/{reference_view.image_name}_inpaint.png")
        torchvision.utils.save_image(inpaint_rgb - rgb, f"tmp/{reference_view.image_name}_diff.png")
        torchvision.utils.save_image(torch.tensor(unseen_mask, dtype=torch.float), f"tmp/{reference_view.image_name}_mask.png")
        
        estimated_depth = estimate_depth_depth_anything_v2(inpaint_rgb)
    
       
    # Solution1. Align the depth using scale and shift, aquried minimizing the L1 loss
    # depth_ref = align_depth_gradient_descent(depth, estimated_depth, unseen_mask, tb_writer)
    # Solution2. Align the depth using scale and shift, aquired by least square method
    # depth_ref = align_depth_least_square(depth, estimated_depth, unseen_mask)
    # Solution3. Align the depth using poisson blending
    depth_ref = align_depth_poisson(depth, estimated_depth, unseen_mask)
    # depth_ref = align_depth_poisson(depth_ref * unseen_mask + depth[0] * (1 - unseen_mask), estimated_depth, unseen_mask)
    
    
    with torch.no_grad():
        # reference_view_aligned_depth = depth_ref * unseen_mask + depth[0] * (1 - unseen_mask)
        reference_view_aligned_depth = depth_ref
        
        
        
        initial_gs_xyz, initial_gs_rgb = unproject(reference_view, reference_view_aligned_depth, inpaint_rgb, device="cuda", mask=unseen_mask)
        initial_gs_xyz, initial_gs_rgb = voxel_downsample(initial_gs_xyz, initial_gs_rgb, voxel_down_size)

    
    # fix some gaussians
    # gaussians.inpaint_setup(opt) # random init gaussians for inpainting
    gaussians.inpaint_setup(opt, initial_gs_xyz, initial_gs_rgb)
    
    # save composed gaussians
    with torch.no_grad():
        point_cloud_path = os.path.join(model_path, "point_cloud/object_inpaint_init")
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # tensorboard
        
        depth_map = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
        tb_writer.add_images('depth_alignment/depth', depth_map[None], global_step=0)
        
        disparity_map = colormap(1 / (depth / depth.max()).cpu().numpy()[0], cmap='turbo')
        tb_writer.add_images('depth_alignment/disparity', disparity_map[None], global_step=0)
        
        
        
        estiamted_disparity_map = colormap((estimated_depth).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/estimated_disparity', estiamted_disparity_map[None], global_step=0)
        
    
        # estimated_real_depth = 1 / (estimated_depth + 1e-6)    
        # estiamted_depth_map = colormap((estimated_real_depth).cpu().numpy(), cmap='turbo')
        # tb_writer.add_images('depth_alignment/estimated_depth', estiamted_depth_map[None], global_step=0)
        
        aligned_depth_map = colormap(1 / (reference_view_aligned_depth / reference_view_aligned_depth.max()).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/aligned_disparity', aligned_depth_map[None], global_step=0)
        
        aligned_depth_map = colormap((reference_view_aligned_depth / reference_view_aligned_depth.max()).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/aligned_depth', aligned_depth_map[None], global_step=0)



    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()
    def mask_to_bbox(mask):
        # Find the rows and columns where the mask is non-zero
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
    
        return xmin, ymin, xmax, ymax

    def crop_using_bbox(image, bbox):
        xmin, ymin, xmax, ymax = bbox
        return image[:, ymin:ymax+1, xmin:xmax+1]

    def divide_into_patches(image, K):
        B, C, H, W = image.shape
        patch_h, patch_w = H // K, W // K
        patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
        patches = patches.view(B, C, patch_h, patch_w, -1)
        return patches.permute(0, 4, 1, 2, 3)

    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    
    
    
    
    for iteration in range(1, iterations + 1):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        # torchvision.utils.save_image(image, f"tmp/render.png")
        
        # finetune only masked region: 1. LPIPS loss
        loss = torch.tensor(0.0).cuda()
        
        kernel_size = 10
        image_mask = cv2.dilate(viewpoint_cam.original_image_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
        image_mask_tensor = torch.tensor(image_mask).cuda().repeat(3,1,1)
        image_m = image * image_mask_tensor
        gt_image_m = gt_image * image_mask_tensor
        Ll1 = torch.tensor(0.0).cuda()
        # masked
        loss += (1.0 - opt.lambda_dssim) * l1_loss(image_m, gt_image_m) + opt.lambda_dssim * (1 - ssim(image_m, gt_image_m))
        
        # unmasked region
        image_um = image * (1 - image_mask_tensor)
        gt_image_um = gt_image * (1 - image_mask_tensor)
        loss += 0.8 * l1_loss(image_um, gt_image_um) + 0.2 * (1 - ssim(image_um, gt_image_um))
        
        
        
        
        
        
        
        try: # To catch the error when the mask is all zeros
            bbox = mask_to_bbox(image_mask_tensor[0])
            cropped_image = crop_using_bbox(image, bbox)
            cropped_gt_image = crop_using_bbox(gt_image, bbox)
            K = 2
            rendering_patches = divide_into_patches(cropped_image[None, ...], K)
            gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
            lpips_loss = LPIPS((rendering_patches.squeeze()*2-1), (gt_patches.squeeze()*2-1)).mean()
            loss += opt.lambda_lpips * lpips_loss
        except:
            pass
        
        # fintune only masked region: 2. normal/dist regularization
        lambda_normal = opt.lambda_normal if iteration > 700 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 300 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal'] * image_mask_tensor
        surf_normal = render_pkg['surf_normal'] * image_mask_tensor
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        
        
        
        total_loss = loss + dist_loss + normal_loss
        
        # breakpoint()
        
        
        
        total_loss.backward()
        
        
        
        
        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % 300 == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            if (iteration == iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_inpaint.pth")
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}_object_inpaint".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians

def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, voxel_down_size : float,  finetune_iteration: int):
    # 1. load gaussian checkpoint
    dataset.stage = 'inpaint'
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    # 2. inpaint selected object
    gaussians = finetune_inpaint(scene, opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, None, select_obj_id, scene.cameras_extent, voxel_down_size, finetune_iteration)
    
    # 3. render new result
    scene = Scene(dataset, gaussians, load_iteration=str(finetune_iteration)+'_object_inpaint', shuffle=False)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    train_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'train',  "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'test', "ours_{}".format(scene.loaded_iter))
    
    with torch.no_grad():
        if not skip_train:
            print("export removal training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)
             
            # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test and (len(scene.getTestCameras()) > 0):
            print("export removal rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
            
        if args.render_path:
            print("render videos ...")
            traj_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'traj', "ours_{}".format(scene.loaded_iter))
            os.makedirs(traj_dir, exist_ok=True)
            n_fames = 240
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(base_dir=traj_dir,
                        input_dir=traj_dir, 
                        out_name='render_traj', 
                        num_frames=n_fames)

        if not args.skip_mesh:
            print("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(scene.getTrainCameras())
            # extract the mesh and save
            if args.unbounded:
                name = 'fuse_unbounded.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = 'fuse.ply'
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--voxel_down_size", default=0.1, type=float, help='voxel size for voxel downsampling')
    parser.add_argument("--finetune_iteration", default=10000, type=int, help='Inpaint: number of finetune iterations')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    
    iteration = str(iteration) + "_object_removal"
    
    # inpaint
    inpaint(dataset, iteration, pipe, args.skip_train, args.skip_test, opt.extract(args), 0, args.voxel_down_size, args.finetune_iteration)