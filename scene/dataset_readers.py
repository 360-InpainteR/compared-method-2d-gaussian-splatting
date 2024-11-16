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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    image_mask: str
    depth: np.array
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(
    cam_extrinsics, cam_intrinsics, images_folder, stage, test_images_folder
):
    train_cam_infos = []
    test_cam_infos = []
    # # handle for inpaint stage
    # if stage == "inpaint":
    #     # new of reference image is corresponding to the last image in the training cam
    #     # replace the dir of image_folder to the dir name "reference"    
    #     reference_img_path = images_folder.replace(f"{os.path.basename(images_folder)}", "reference")
    #     breakpoint()
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {} - extrinsics {}".format(idx + 1, len(cam_extrinsics))
        )
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        train_image_path = os.path.join(images_folder, os.path.basename(extr.name))
        test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name))
        # handle for inpaint_images
        if stage == "inpaint":
            image_extensions = [".jpg", ".JPG", ".png", ".PNG"]
            for ext in image_extensions:
                train_image_path = os.path.join(images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(train_image_path):
                    break
            for ext in image_extensions:
                test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(test_image_path):
                    break
        # check if image exists in training image folder
        if os.path.exists(train_image_path):
            image_path = train_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            image_extensions = [".jpg", ".JPG", ".png", ".PNG"]
            if stage == "inpaint":
                # image_base_path = os.path.join(os.path.dirname(images_folder)+'/inpaint_2d_unseen_mask_great', os.path.splitext(os.path.basename(extr.name))[0])
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
                # image_base_path = os.path.join(os.path.dirname(images_folder)+'/inpaint_2d_unseen_mask', f"{idx:05d}")
            elif stage == "train":
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
            elif stage == "removal":
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/rend_object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
            else:
                raise ValueError(f"stage {stage} not supported")

            image_mask = None
            image_mask_path = None
            for ext in image_extensions:
                if os.path.exists(image_base_path + ext):
                    image_mask_path = image_base_path + ext
                    break
            
            if stage == "removal":
                image_mask = np.array(Image.open(image_mask_path).convert("L"))
                mask_array = np.where(image_mask > 10, 1, 0)
                image_mask = Image.fromarray((mask_array * 255).astype(np.uint8))
            else:
                image_mask = np.array(Image.open(image_mask_path).convert("L"))
                mask_array = np.where(image_mask > 127, 1, 0)
                image_mask = Image.fromarray((mask_array * 255).astype(np.uint8))

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=image_mask,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            train_cam_infos.append(cam_info)
        # check if image exists in testing image folder
        elif os.path.exists(test_image_path):
            image_path = test_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=None,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            test_cam_infos.append(cam_info)
        else:
            # raise ValueError(f"Image: {image_name} not found in train / test")
            print(f"Image: {extr.name} not found in train / test")
            continue

    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, stage="inpaint"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # load training images from {images}, and testing images from {test_images}
    reading_dir = "images" if images == None else images
    train_cam_infos_unsorted, test_cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        stage=stage,
        test_images_folder=os.path.join(path, "test_images"),
    )

    train_cam_infos = sorted(
        train_cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )
    test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapCamerasAura(
    cam_extrinsics, cam_intrinsics, images_folder, stage, test_images_folder
):
    train_cam_infos = []
    test_cam_infos = []
    # # handle for inpaint stage
    # if stage == "inpaint":
    #     # new of reference image is corresponding to the last image in the training cam
    #     # replace the dir of image_folder to the dir name "reference"    
    #     reference_img_path = images_folder.replace(f"{os.path.basename(images_folder)}", "reference")
    #     breakpoint()
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {} - extrinsics {}".format(idx + 1, len(cam_extrinsics))
        )
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        train_image_path = os.path.join(images_folder, os.path.basename(extr.name))
        test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name))
        # handle for inpaint_images
        if stage == "inpaint":
            image_extensions = [".jpg", ".JPG", ".png", ".PNG"]
            for ext in image_extensions:
                train_image_path = os.path.join(images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(train_image_path):
                    break
            for ext in image_extensions:
                test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(test_image_path):
                    break
        # check if image exists in training image folder
        if os.path.exists(train_image_path):
            image_path = train_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            image_extensions = [".jpg", ".JPG", ".png", ".PNG"]
            if stage == "inpaint":
                # image_base_path = os.path.join(os.path.dirname(images_folder)+'/inpaint_2d_unseen_mask_great', os.path.splitext(os.path.basename(extr.name))[0])
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
                # image_base_path = os.path.join(os.path.dirname(images_folder)+'/inpaint_2d_unseen_mask', f"{idx:05d}")
            elif stage == "train":
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
            elif stage == "removal":
                image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/rend_object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
            else:
                raise ValueError(f"stage {stage} not supported")

            image_mask = None
            image_mask_path = None
            for ext in image_extensions:
                if os.path.exists(image_base_path + ext):
                    image_mask_path = image_base_path + ext
                    break

            if stage == "removal":
                image_mask = np.array(Image.open(image_mask_path).convert("L"))
                mask_array = np.where(image_mask > 10, 1, 0)
                image_mask = Image.fromarray((mask_array * 255).astype(np.uint8))
            else:
                image_mask = np.array(Image.open(image_mask_path).convert("L"))
                mask_array = np.where(image_mask > 127, 1, 0)
                image_mask = Image.fromarray((mask_array * 255).astype(np.uint8))

            image_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/object_masks",
                    os.path.splitext(os.path.basename(extr.name))[0],
                )
            
            depths_base_path = os.path.join(
                    os.path.dirname(images_folder) + "/depths",
                    os.path.splitext(os.path.basename(extr.name))[0],
            )
            for ext in image_extensions:
                if os.path.exists(depths_base_path + ext):
                    depth_path = depths_base_path + ext
                    break
                
            depth = np.array(Image.open(depth_path), dtype=np.float32)
            depth = np.mean(depth, axis=2, keepdims=True)
            # divide by 255 to get depth in [0, 1]
            depth = depth / 255.0

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=image_mask,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                depth=depth
            )
            train_cam_infos.append(cam_info)
        # check if image exists in testing image folder
        elif os.path.exists(test_image_path):
            image_path = test_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=None,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                depth=None
            )
            test_cam_infos.append(cam_info)
        else:
            # raise ValueError(f"Image: {image_name} not found in train / test")
            print(f"Image: {extr.name} not found in train / test")
            continue

    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos


def readColmapSceneInfoAura(path, images, eval, llffhold=8, stage="inpaint"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # load training images from {images}, and testing images from {test_images}
    reading_dir = "images" if images == None else images
    train_cam_infos_unsorted, test_cam_infos_unsorted = readColmapCamerasAura(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        stage=stage,
        test_images_folder=os.path.join(path, "test_images"),
    )

    train_cam_infos = sorted(
        train_cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )
    test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Aura": readColmapSceneInfoAura,
}