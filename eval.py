import argparse
import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from models.decom import CTDN
import utils

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_model(ckpt_path, device):
    """加载训练好的 CTDN 模型"""
    model = CTDN(channels=64)
    checkpoint = utils.logging.load_checkpoint(ckpt_path, device)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_image_paths(input_path):
    """解析输入路径：单张图片或文件夹"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for ext in IMG_EXTENSIONS:
            paths.extend(sorted(glob.glob(os.path.join(input_path, f'*{ext}'))))
            paths.extend(sorted(glob.glob(os.path.join(input_path, f'*{ext.upper()}'))))
        return paths
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")


def decompose_image(model, image_tensor, device):
    """
    对单张图像做分解
    Args:
        image_tensor: [1, 3, H, W] 低光图
    Returns:
        low_R, low_L, low_fea  (均为 [1, 3, H/8, W/8], 特征空间)
    """
    with torch.no_grad():
        x = torch.cat([image_tensor, image_tensor], dim=1).to(device)
        output = model(x, pred_fea=None)
        low_R = output["low_R"]
        low_L = output["low_L"]
        low_fea = output["low_fea"]
    return low_R, low_L, low_fea


def process_one(model, image_path, output_dir, device):
    """处理单张图片并保存结果"""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    low_R, low_L, low_fea = decompose_image(model, img_tensor, device)

    low_R_up = F.interpolate(low_R, size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False)
    low_L_up = F.interpolate(low_L, size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False)
    low_fea_up = F.interpolate(low_fea, size=(h + pad_h, w + pad_w), mode='bilinear', align_corners=False)

    low_R_up = low_R_up[:, :, :h, :w]
    low_L_up = low_L_up[:, :, :h, :w]
    low_fea_up = low_fea_up[:, :, :h, :w]

    low_L_gray = low_L_up[:, 0:1, :, :]

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    utils.logging.save_image(img_tensor[:, :, :h, :w],
                             os.path.join(output_dir, f'{basename}_input.png'))
    utils.logging.save_image(low_fea_up,
                             os.path.join(output_dir, f'{basename}_fea.png'))
    utils.logging.save_image(low_R_up,
                             os.path.join(output_dir, f'{basename}_R.png'))
    utils.logging.save_image(low_L_gray.repeat(1, 3, 1, 1),
                             os.path.join(output_dir, f'{basename}_L.png'))

    return basename


def main():
    parser = argparse.ArgumentParser(description='CTDN Stage1 推理: 分解低光图')
    parser.add_argument('--input_path', required=True, help='低光图像路径 或 文件夹路径')
    parser.add_argument('--ckpt', default='ckpt/stage1/stage1_weight.pth.tar', help='模型权重路径')
    parser.add_argument('--output_dir', default='results/stage1_infer/', help='输出目录')
    parser.add_argument('--device', default='cuda', help='设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"=> 设备: {device}")

    # --- 加载模型（只加载一次） ---
    print(f"=> 加载模型: {args.ckpt}")
    model = load_model(args.ckpt, device)

    # --- 解析输入路径 ---
    image_paths = get_image_paths(args.input_path)
    print(f"=> 共检测到 {len(image_paths)} 张图像")

    # --- 逐张处理 ---
    for i, img_path in enumerate(image_paths):
        name = process_one(model, img_path, args.output_dir, device)
        print(f"   [{i+1}/{len(image_paths)}] {name} 完成")

    print(f"\n=> 全部完成，结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
