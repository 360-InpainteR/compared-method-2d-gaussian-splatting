import cv2
import torch
from pietorch import blend, blend_wide

from utils.general_utils import vis_depth
from utils.loss_utils import l1_loss
from tqdm import tqdm


midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas = midas.to("cuda")
midas.eval()
for param in midas.parameters():
    param.requires_grad = False
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img =  (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)
    
    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    # prediction_map = vis_depth((1 / prediction).detach().cpu().numpy())
    
    # if DEBUG:
    # cv2.imwrite("tmp/estimated_depth_i.png", prediction_map)
        
    # The output depth is disparity, not depth    
    return prediction
    


# zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to("cuda")
# def estimate_depth(img):
#     from PIL import Image
#     # image = Image.open("tmp/original_img_i.png")
#     # x = pil_to_batched_tensor(image).to("cuda")
#     # # depth = zoe.infer(x)
    
#     # depth = zoe.infer_pil(image, output_type='tensor')
#     # breakpoint()
#     depth = zoe.infer(img.unsqueeze(0))[0]
#     return depth
    
    
#     # if debug_utils.DEBUG:
#     #     depth_map = vis_depth(depth[0].detach().cpu().numpy())
#     #     cv2.imwrite("tmp/estimated_depth_i.png", depth_map)
        
#     # return depth

from transformers import pipeline
from PIL import Image
def estimate_depth_depth_anything_v2(rgb):
    """_summary_

    Args:
        rgb (torch.Tensor): _description_

    Returns:
        torch.Tensor: esitmated depth
    """
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
    rgb_numpy = rgb.permute(1, 2, 0).detach().cpu().numpy()
    pil_image = Image.fromarray((rgb_numpy * 255).astype("uint8"))
    # image = Image.open(refernece_img_path)
    depth = pipe(pil_image)["predicted_depth"] # torch.Tensor (1, H, W)
    # resize depth to the same size as the input image
    depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(rgb.shape[1], rgb.shape[2]), mode="bicubic", align_corners=False).squeeze()
    
    
    
    return depth.cuda()


corner = torch.tensor([0, 0], dtype=torch.int)
@torch.no_grad()
def align_depth_poisson(depth, estimated_depth, mask):
    # estimated_depth = (estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min())
    estimated_depth = 1 / (estimated_depth + 1e-8)
    
    
    
    
    # aligned_depth = blend(depth.repeat(3, 1, 1), estimated_depth.unsqueeze(0).repeat(3, 1, 1), mask, corner, mix_gradients=False, channels_dim=0)
    aligned_depth = blend_wide(depth.repeat(3, 1, 1), estimated_depth.unsqueeze(0).repeat(3, 1, 1), mask, corner, mix_gradients=False, channels_dim=0)
    
    # aligned_depth = blend(depth, estimated_depth.unsqueeze(0), mask, corner, mix_gradients=False, channels_dim=0)
    # aligned_depth = blend_wide(depth, estimated_depth.unsqueeze(0), mask, corner, mix_gradients=False, channels_dim=0)
    # aligned_depth = blend_wide(depth.squeeze(0), estimated_depth, mask, corner, mix_gradients=False)
    
    aligned_depth = torch.clamp(aligned_depth, 0.01, 100)
    
    # aligned_depth_map = vis_depth(aligned_depth[0].detach().cpu().numpy())
    
    # if debug_utils.DEBUG:
    # cv2.imwrite("tmp/aligned_depth_i.png", aligned_depth_map)
    return aligned_depth[0]


def align_depth_gradient_descent(depth, estimated_depth, unseen_mask, tb_writer=None):
    """Align depth using gradient descent

    Args:
        depth (torch.Tensor): real depth (1, H, W)
        estimated_depth (_type_): estimated depth (H, W)
        unseen_mask (_type_): unseen mask (H, W)

    Returns:
        torch.Tensor: Aligned depth
    """
    
    
    
    depth_scale = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, dtype=torch.float, device="cuda"))
    depth_shift = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True, dtype=torch.float, device="cuda"))
    
    progress_bar = tqdm(range(0, 1000), desc="Fine-tune Monocular Depth")
    # estimated_depth = -(estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min()).detach()
    # estimated_depth = (estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min())
    estimated_depth = 1 / (estimated_depth + 1e-6)
    
    # estimated_depth_normalized = (estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min())
    # depth_normalized = (depth - depth[unseen_mask==0].min()) / (depth[unseen_mask==0].max() - depth[unseen_mask==1].min())
    

    depth_optimizer = torch.optim.Adam([depth_scale, depth_shift], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(depth_optimizer, step_size=500, gamma=0.1, last_epoch=-1)
    
    
    for iteration in range(1000):
        depth_optimizer.zero_grad()
        # depth_ref = (estimated_depth + depth_shift) * depth_scale
        depth_ref = estimated_depth * depth_scale + depth_shift
        depth_ref = torch.clamp(depth_ref, 0.01, 100)
        
        Ldepth = l1_loss(depth_ref * (1 - unseen_mask), depth[0].detach() * (1 - unseen_mask))
        Ldepth.backward()
        
        # torch.nn.utils.clip_grad_norm_([depth_scale, depth_shift], max_norm=1.0)
        depth_optimizer.step()
        scheduler.step()
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{Ldepth:.{7}f}"})
            progress_bar.update(1)
            tb_writer.add_scalar("depth_alignment/Ldepth", Ldepth.item(), iteration)
    progress_bar.close()
    print(f"##### Depth Scale: {depth_scale.item()}, Depth Shift: {depth_shift.item()} #####")
    print(f"##### loss: {Ldepth.item()} #####")
    return depth_ref

@torch.no_grad()
def align_depth_least_square(depth, estimated_depth, unseen_mask):
    """Align depth using least square

    Args:
        depth (torch.Tensor): real depth (1, H, W)
        estimated_deth (_type_): estimated depth (H, W)
        unseen_mask (_type_): unseen mask (H, W)

    Returns:
        torch.Tensor: Aligned depth
    """
    estimated_depth = 1 / (estimated_depth + 1e-8)
    
    unmask_real_depth_flat = depth[0][unseen_mask == 0].view(-1)
    unmask_estimated_depth_flat = estimated_depth[unseen_mask == 0].view(-1)
    A = torch.stack([unmask_estimated_depth_flat, torch.ones_like(unmask_estimated_depth_flat)], dim=1)
    B = unmask_real_depth_flat.unsqueeze(1)
    X = torch.linalg.lstsq(A, B)
    
    scale, shift = X.solution[0, 0], X.solution[1, 0]

    depth_ref = (scale * estimated_depth + shift)
    # depth_ref = (scale * estimated_depth + shift).clamp(0.01, 100)
    
    return depth_ref
