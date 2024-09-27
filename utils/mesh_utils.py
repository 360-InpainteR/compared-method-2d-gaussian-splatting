#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8, save_gray_img_u8
from gaussian_renderer import render_mask
from functools import partial
import open3d as o3d
import trimesh
from utils.warping_utils import unproject, warping
import torchvision

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.render_mask = partial(render_mask, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        self.maskmaps = []
        self.seenmaps = []
        self.unseenmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, gen_unseen_mask=False):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        
        # # TODO: Warping
        # ids = np.random.choice(len(viewpoint_stack), 4)
        # ref_viewpoint_cams = [viewpoint_stack[i] for i in ids]
        # ref_viewpoint_cams = [viewpoint_stack[10]]
        
        
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            # render_pkg = self.render(viewpoint_cam, self.gaussians, optimize_is_masked=True)
            
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            
            if gen_unseen_mask:
                # TODO: generate unseen mask
                object_mask = torch.tensor(viewpoint_cam.original_image_mask, dtype=torch.float).cuda()
                object_mask = object_mask.repeat(3, 1, 1)
                unseen_mask_final = torch.zeros_like(object_mask)
                unseen_mask = torch.zeros_like(object_mask)
                ref_viewpoint_cams = [viewpoint_stack[id] for id in range(len(viewpoint_stack)) if id != i]
                for ref_viewpoint_cam in ref_viewpoint_cams:
                    # SOL.1 Warp the object mask
                    # render_pkg = render(ref_viewpoint_cam, gaussians)
                    # ref_depth = render_pkg['surf_depth']
                    ref_object_mask = torch.tensor(ref_viewpoint_cam.original_image_mask, dtype=torch.float).cuda()    
                
                    ref_object_mask_warp, original_indices, ref_indices = warping(viewpoint_cam, ref_viewpoint_cam, object_mask, depth, device='cuda')
                        
                    AND_ref = ref_object_mask * ref_object_mask_warp[:1, :]
                    # original_ref_object_mask = object_mask.clone()
                    unseen_mask[:, original_indices[:, 0], original_indices[:, 1]] = AND_ref[:, ref_indices[:, 0], ref_indices[:, 1]]
                    unseen_mask = unseen_mask * object_mask
                    unseen_mask_final += unseen_mask
                    
                    # torchvision.utils.save_image(object_mask, f"tmp3/a_object_mask.png")
                    # torchvision.utils.save_image(ref_object_mask, f"tmp3/a_ref_object_mask.png")
                    # torchvision.utils.save_image(ref_object_mask_warp[:1, :], "tmp3/a_ref_vierw_object_mask_warp.png")
                    # torchvision.utils.save_image(unseen_mask[:1, :], "tmp3/a_unseen_mask.png")
                    
                    
                    # # SOL.2 Warp the depth
                    # ref_object_mask = torch.tensor(ref_viewpoint_cam.original_image_mask, dtype=torch.float).cuda()    
                    # ref_depth_warp, original_indices, ref_indices = warping(viewpoint_cam, ref_viewpoint_cam, depth, depth, device='cuda')
                    # ref_depth = self.render(ref_viewpoint_cam, self.gaussians)['surf_depth']
                    # AND_ref = ref_object_mask[:1, :] * ((ref_depth_warp - ref_depth).abs() < 1.0).float()
                    # unseen_mask[:, original_indices[:, 0], original_indices[:, 1]] = AND_ref[:, ref_indices[:, 0], ref_indices[:, 1]]
                    # unseen_mask = unseen_mask * object_mask
                    # unseen_mask_final += unseen_mask
                    
                    # # torchvision.utils.save_image(ref_object_mask, f"tmp3/a_ref_object_mask.png")
                    # # torchvision.utils.save_image(AND_ref, f"tmp3/a_ref_object_mask_warp.png")
                    # # torchvision.utils.save_image(unseen_mask[:1, :], "tmp3/a_unseen_mask.png")
                    # # breakpoint()
                    
                    
                # TYPE1: gen by warp
                unseen_mask_final = unseen_mask_final / len(ref_viewpoint_cams)
                # TYPE2: no gaussian region
                unseen_mask_final[:1][depth == 0] = 1 # unseen region 
                
                # Thresholding
                
                thr = 0.35
                unseen_mask_final[unseen_mask_final > thr] = 1
                unseen_mask_final[unseen_mask_final <= thr] = 0
                
                # torchvision.utils.save_image(object_mask, f"tmp3/a_object_mask.png")
                # torchvision.utils.save_image(unseen_mask_final[:1, :], "tmp3/a_unseen_mask_final.png")
                self.unseenmaps.append(unseen_mask_final[:1, :].cpu())
                # breakpoint()
            
            
            try:
                render_pkg = self.render_mask(viewpoint_cam, self.gaussians)
                is_masked = render_pkg['is_masked'][0][None]
                seen = render_pkg['is_masked'][1][None]
                self.maskmaps.append(is_masked.cpu())
                self.seenmaps.append(seen.cpu())
                
                # # TODO: Warping
                # # unrpoject the unseen to 3D space for initialize gaussians
                # # seen > 0.5 to 1, else 0
                # seen_01 = (seen > 0.5).float().repeat(3, 1, 1)
                # # unproject the depth to 3D
                
                # for ref_viewpoint_cam in ref_viewpoint_cams:
                #     render_pkg = self.render(ref_viewpoint_cam, self.gaussians)
                #     ref_depth = render_pkg['surf_depth']
                    
                #     ref_render_pkg = self.render_mask(ref_viewpoint_cam, self.gaussians)
                #     ref_seen = ref_render_pkg['is_masked'][1][None]
                #     ref_seen_01 = (ref_seen > 0.5).float().repeat(3, 1, 1)
                    
                #     seen_01_warp = warping(ref_viewpoint_cam, viewpoint_cam, ref_seen_01, ref_depth, device='cuda')
                    
                    
                #     torchvision.utils.save_image(seen_01, f"tmp2/{viewpoint_cam.image_name}_seen.png")
                #     torchvision.utils.save_image(seen_01_warp[:1, :], f"tmp2/{viewpoint_cam.image_name}_seen_warp.png")

            except Exception as e:           
                pass
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.maskmaps = torch.stack(self.maskmaps, dim=0)
        # self.seenmaps = torch.stack(self.seenmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        depth_normal_path = os.path.join(path, "depth_normal")
        mask_path = os.path.join(path, "object_mask")
        seen_path = os.path.join(path, "seen_mask")
        unseen_path = os.path.join(path, "unseen_mask")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(depth_normal_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(seen_path, exist_ok=True)
        os.makedirs(unseen_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_gray_img_u8(self.maskmaps[idx][0].cpu().numpy(), os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
            save_gray_img_u8(self.seenmaps[idx][0].cpu().numpy(), os.path.join(seen_path, '{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(depth_normal_path, '{0:05d}'.format(idx) + ".png"))
            save_gray_img_u8(self.unseenmaps[idx][0].cpu().numpy(), os.path.join(unseen_path, '{0:05d}'.format(idx) + ".png"))