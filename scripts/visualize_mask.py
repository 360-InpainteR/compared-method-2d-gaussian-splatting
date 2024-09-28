import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import re
from natsort import natsorted


def blend_image_with_mask(image, mask, alpha=0.5, obj_id=None):
    """
    Blends an image with a colormap color where the mask is 255 and keeps the original image where the mask is 0.

    Parameters:
    - image: The original image as a numpy array (H, W, 3).
    - mask: The binary mask image (H, W) with values 0 and 255.
    - alpha: The opacity level of the color in the masked area (0 = fully transparent, 1 = fully opaque).
    - obj_id: Index for the colormap color.

    Returns:
    - Blended image as a numpy array.
    """
    # Ensure the mask is a 2D array
    mask = mask.astype(np.uint8)

    # Get the color from the colormap based on the obj_id
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array(cmap(cmap_idx)[:3])  # Extract the RGB values from the colormap

    # Normalize the mask to range [0, 1] for blending
    mask_normalized = mask / 255.0

    # Create a color overlay of the same shape as the image
    color_overlay = (
        np.ones_like(image, dtype=np.float32) * color * 255
    )  # Scale to 0-255

    # Blend the color overlay with the original image based on the mask
    blended_image = image * (1 - mask_normalized[..., None] * alpha) + color_overlay * (
        mask_normalized[..., None] * alpha
    )

    return blended_image


def render_video_with_mask(
    img_dir, mask_dir, output_video, alpha=0.5, fps=30, obj_id=None
):
    """
    Renders a video by blending images with corresponding masks using a colormap color in masked areas.

    Parameters:
    - img_dir: Directory containing the input images.
    - mask_dir: Directory containing the corresponding masks.
    - output_video: Path to the output video file.
    - alpha: Opacity of the color in the masked areas.
    - fps: Frames per second for the output video.
    - obj_id: Index for the colormap color.
    """
    # Get sorted list of images and masks
    img_files = natsorted(
        [
            f
            for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg", ".jpeg", "PNG", "JPG", "JPEG"))
        ]
    )
    mask_files = natsorted(
        [
            f
            for f in os.listdir(mask_dir)
            if f.endswith((".png", ".jpg", ".jpeg", "PNG", "JPG", "JPEG"))
        ]
    )

    print(f"Found {len(img_files)} images and {len(mask_files)} masks.")

    # Check if the number of images matches the number of masks
    if len(img_files) != len(mask_files):
        print("The number of images and masks do not match.")

        print("Cropping to the shorter one: ", min(len(img_files), len(mask_files)))
        img_files = img_files[: min(len(img_files), len(mask_files))]
        mask_files = mask_files[: min(len(img_files), len(mask_files))]
        # return

    # Read the first image to get the frame size
    first_image_path = os.path.join(img_dir, img_files[0])
    first_image = np.array(Image.open(first_image_path).convert("RGB"))
    height, width, _ = first_image.shape

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate over each image and mask, blend them, and write to the video
    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Read the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(
            Image.open(mask_path).convert("L")
        )  # Convert mask to grayscale (0 and 255)

        # Blend the image with the mask using the colormap color
        blended_image = blend_image_with_mask(image, mask, alpha, obj_id=obj_id)

        # Convert to the right format for OpenCV
        blended_image_bgr = cv2.cvtColor(
            blended_image.astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        # Write the frame to the video
        video_writer.write(blended_image_bgr)

        print(f"Processed {img_file} with {mask_file}")

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    render_video_with_mask(
        args.img_dir, args.mask_dir, f"{args.output_dir}/vis_mask.mp4"
    )
