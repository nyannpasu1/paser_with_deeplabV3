import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from omegaconf import OmegaConf
from models.deeplabv3 import DeepLabV3
import utils
import os
import cv2
import matplotlib.pyplot as plt

# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description='DeepLabV3 Segmentation Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save predicted mask')
    parser.add_argument('--input_dir', type=str, help='Directory of images for batch prediction')
    parser.add_argument('--output_dir', type=str, help='Directory to save predicted masks')
    return parser.parse_args()

def load_image(img_path, n_channels):
    img = Image.open(img_path).convert('L' if n_channels == 1 else 'RGB')
    img = transforms.ToTensor()(img)
    if n_channels == 1:
        img = img.unsqueeze(0) if img.ndim == 2 else img
    return img.unsqueeze(0)  # shape: (1, C, H, W)

def apply_mask_on_image(img, mask, alpha=0.5):
    # img: HWC, uint8; mask: HW, int
    color_mask = np.zeros_like(img)
    palette = [
        [0, 0, 0],      # 类别0:黑
        [255, 0, 0],    # 类别1:红
        [0, 255, 0],    # 类别2:绿
        [0, 0, 255],    # 类别3:蓝
    ]
    for i, color in enumerate(palette):
        color_mask[mask == i] = color
    return cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

def batch_predict(model, cfg, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for fname in img_files:
        img_path = os.path.join(input_dir, fname)
        img = load_image(img_path, cfg.model.n_channels).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # 保存mask
        mask_img = Image.fromarray(pred * (255 // max(1, cfg.model.n_classes-1)))
        mask_img.save(os.path.join(output_dir, fname))
        # 拼接原图和mask
        orig = Image.open(img_path).convert('RGB')
        mask_vis = mask_img.convert('RGB').resize(orig.size)
        concat = Image.new('RGB', (orig.width + mask_vis.width, orig.height))
        concat.paste(orig, (0, 0))
        concat.paste(mask_vis, (orig.width, 0))
        concat_name = os.path.splitext(fname)[0] + '_concat.png'
        concat.save(os.path.join(output_dir, concat_name))
        # 彩色overlay
        orig_np = np.array(orig)
        overlay = apply_mask_on_image(orig_np, pred)
        overlay_img = Image.fromarray(overlay)
        overlay_name = os.path.splitext(fname)[0] + '_overlay.png'
        overlay_img.save(os.path.join(output_dir, overlay_name))
        print(f"Saved: {os.path.join(output_dir, fname)}, {os.path.join(output_dir, concat_name)}, {os.path.join(output_dir, overlay_name)}")

def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device('cpu')
    # 构建模型
    model = DeepLabV3(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, backbone=cfg.model.backbone, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # 批量预测
    if args.input_dir and args.output_dir:
        batch_predict(model, cfg, args.input_dir, args.output_dir, device)
    elif args.input and args.output:
        img = load_image(args.input, cfg.model.n_channels)
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        mask_img = Image.fromarray(pred * (255 // max(1, cfg.model.n_classes-1)))
        mask_img.save(args.output)
        print(f"Saved predicted mask to {args.output}")
    else:
        print("Please provide either --input/--output or --input_dir/--output_dir.")

if __name__ == '__main__':
    main()
