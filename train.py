
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import datasets
import utils
from models.decom import CTDN


# ===================== 参数解析 =====================
def dict2namespace(config_dict):
    """将 dict 递归转换为 argparse.Namespace (兼容现有代码风格)"""
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Stage1: CTDN Retinex Decomposition Training')
    parser.add_argument("--config", default='stage1.yml', type=str,
                        help="配置文件路径 (configs/ 目录下)")
    parser.add_argument('--resume', default='', type=str,
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument("--image_folder", default='results/stage1/', type=str,
                        help="验证图像保存路径")
    parser.add_argument('--seed', default=230, type=int,
                        help='随机种子')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)

    config = dict2namespace(config_dict)
    return args, config


# ===================== 光照平滑损失 (TV Loss) =====================
def illumination_smooth_loss(illumination, image):
    """
    光照图平滑损失: 用原图梯度加权，边缘处允许光照突变
    
    公式: Σ |∇L| * exp(-λ * |∇I|)
    意思是: 原图梯度大的地方(边缘)，允许光照也有梯度
            原图梯度小的地方(平滑区)，惩罚光照的梯度
    """
    batch, channel, height, width = illumination.shape

    # 对光照图求梯度
    grad_l_x = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
    grad_l_y = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])

    # 对原图求梯度 (用均值图作为引导)
    if image.shape[1] == 3:
        gray_image = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
    else:
        gray_image = image

    grad_i_x = torch.abs(gray_image[:, :, :, 1:] - gray_image[:, :, :, :-1])
    grad_i_y = torch.abs(gray_image[:, :, 1:, :] - gray_image[:, :, :-1, :])

    # 加权: 原图梯度大时，exp(-10*|∇I|) → 0，允许光照突变
    weight_x = torch.exp(-10.0 * grad_i_x)
    weight_y = torch.exp(-10.0 * grad_i_y)

    smooth_loss_x = (grad_l_x * weight_x).mean()
    smooth_loss_y = (grad_l_y * weight_y).mean()

    return smooth_loss_x + smooth_loss_y


# ===================== 训练主函数 =====================
def main():
    args, config = parse_args_and_config()

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    print(f"=> 使用设备: {device}")

    # --- 随机种子 ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # --- 数据集 ---
    print(f"=> 加载数据集: '{config.data.train_dataset}'")
    DATASET = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = DATASET.get_loaders()
    print(f"   训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")

    # --- 创建模型 ---
    print("=> 创建 CTDN 模型 (channels={})...".format(config.model.channels))
    model = CTDN(channels=config.model.channels)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # --- 优化器 ---
    optimizer = utils.optimize.get_optimizer(config, model.parameters())

    # --- 学习率调度器 ---
    lr_scheduler_type = config.optim.lr_scheduler if hasattr(config.optim, 'lr_scheduler') else 'cosine'
    if lr_scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=config.training.n_epochs,
                                      eta_min=1e-7)
    elif lr_scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # --- 损失函数 ---
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # --- 损失权重 ---
    loss_cfg = config.loss
    w_recon = loss_cfg.weight_recon
    w_ref = loss_cfg.weight_reflectance
    w_smooth = loss_cfg.weight_smooth
    w_illum = loss_cfg.weight_illumination

    # --- 恢复训练 ---
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = utils.logging.load_checkpoint(args.resume, 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('step', 0)
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> 从 checkpoint 恢复: epoch={start_epoch}, step={global_step}")

    # --- 创建保存目录 ---
    os.makedirs(config.data.ckpt_dir, exist_ok=True)

    # ===================== 训练循环 =====================
    print("\n" + "=" * 60)
    print("开始 Stage 1 训练 — CTDN Retinex 分解")
    print("=" * 60)
    print(f"  重建损失权重:      {w_recon}")
    print(f"  反射率一致性权重:  {w_ref}")
    print(f"  光照平滑权重:      {w_smooth}")
    print(f"  光照正则化权重:    {w_illum}")
    print(f"  学习率:            {config.optim.lr}")
    print(f"  训练轮数:          {config.training.n_epochs}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, config.training.n_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_recon = 0.0
        epoch_loss_ref = 0.0
        epoch_loss_smooth = 0.0
        epoch_loss_illum = 0.0
        data_start = time.time()
        data_time = 0.0

        for i, (x, img_ids) in enumerate(train_loader):
            # x: [B, 6, H, W] = [low_img(3ch), high_img(3ch)] 拼接
            # 如果 x 维度为5 (来自某些特殊情况), 展平
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            x = x.to(device)

            # 分离低光图和正常光图
            low_img = x[:, :3, :, :]    # [B, 3, H, W]
            high_img = x[:, 3:, :, :]   # [B, 3, H, W]

            # --- 前向传播: CTDN 分解 ---
            output = model(x, pred_fea=None)  # pred_fea=None → 分解模式

            low_R = output["low_R"]       # 低光反射图 [B, 3, H/8, W/8]
            low_L = output["low_L"]       # 低光光照图 [B, 3, H/8, W/8]
            low_fea = output["low_fea"]   # 低光特征图 [B, 3, H/8, W/8]（CTDN的真正重建目标）
            high_R = output["high_R"]     # 正常光反射图
            high_L = output["high_L"]     # 正常光光照图
            high_fea = output["high_fea"] # 正常光特征图

            # ============================================
            #  1. 特征空间重建: R * L ≈ low_fea（非原图！）
            #     CTDN 在特征空间中做 Retinex 分解，
            #     low_fea 是 channel_down 输出的3通道压缩特征
            # ============================================
            recon_low = l1_loss(low_R * low_L, low_fea)
            recon_high = l1_loss(high_R * high_L, high_fea)
            loss_recon = (recon_low + recon_high) * w_recon

            # ============================================
            #  2. 反射率一致性: low_R ≈ high_R
            #     反射率是物体的固有属性，与光照无关
            # ============================================
            loss_reflectance = l1_loss(low_R, high_R) * w_ref

            # ============================================
            #  3. 光照平滑性: 用特征图梯度引导
            # ============================================
            loss_smooth_low = illumination_smooth_loss(low_L, low_fea)
            loss_smooth_high = illumination_smooth_loss(high_L, high_fea)
            loss_smooth = (loss_smooth_low + loss_smooth_high) * w_smooth

            # ============================================
            #  4. 光照正则化: L 均值 ≈ 特征图亮度
            # ============================================
            low_gray = 0.299 * low_fea[:, 0:1] + 0.587 * low_fea[:, 1:2] + 0.114 * low_fea[:, 2:3]
            high_gray = 0.299 * high_fea[:, 0:1] + 0.587 * high_fea[:, 1:2] + 0.114 * high_fea[:, 2:3]

            loss_illum = l2_loss(low_L[:, 0:1].mean(dim=[1, 2, 3]),
                                 low_gray.mean(dim=[1, 2, 3])) * w_illum

            # ============================================
            #  总损失
            # ============================================
            loss_total = loss_recon + loss_reflectance + loss_smooth + loss_illum

            # --- 反向传播 ---
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防NaN
            optimizer.step()

            global_step += 1
            data_time += time.time() - data_start

            # 累积 epoch 统计
            epoch_loss_total += loss_total.item()
            epoch_loss_recon += loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon
            epoch_loss_ref += loss_reflectance.item() if isinstance(loss_reflectance, torch.Tensor) else loss_reflectance
            epoch_loss_smooth += loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth
            epoch_loss_illum += loss_illum.item() if isinstance(loss_illum, torch.Tensor) else loss_illum

            # --- 每10步打印 ---
            if global_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch[{epoch}] Step[{global_step}] "
                      f"Loss:{loss_total.item():.5f} "
                      f"Recon:{loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon:.4f} "
                      f"Ref:{loss_reflectance.item() if isinstance(loss_reflectance, torch.Tensor) else loss_reflectance:.4f} "
                      f"Smooth:{loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth:.4f} "
                      f"Illum:{loss_illum.item() if isinstance(loss_illum, torch.Tensor) else loss_illum:.4f} "
                      f"LR:{current_lr:.7f} "
                      f"Time:{data_time / (i + 1):.3f}")

            data_start = time.time()

            # --- 验证 & 保存 ---
            if global_step % config.training.validation_freq == 0 and global_step != 0:
                print(f"\n=> 第 {global_step} 步验证...")
                model.eval()
                with torch.no_grad():
                    validate_stage1(model, val_loader, device, args, global_step)

                # 保存 checkpoint
                save_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scheduler is not None:
                    save_dict['scheduler'] = scheduler.state_dict()

                utils.logging.save_checkpoint(
                    save_dict,
                    filename=os.path.join(config.data.ckpt_dir, 'model_latest')
                )

                # 同时保存一份作为 Stage 2 固定加载的权重 (文件名与 ddm.py 中 load_stage1 一致)
                utils.logging.save_checkpoint(
                    save_dict,
                    filename=os.path.join(config.data.ckpt_dir, 'stage1_weight')
                )
                print(f"=> Checkpoint 已保存到 {config.data.ckpt_dir}/\n")

                model.train()

        # --- Epoch 结束: 更新学习率 ---
        if scheduler is not None:
            scheduler.step()

        # --- Epoch 统计 ---
        num_batches = i + 1
        print(f"\n===== Epoch [{epoch}] 完成 =====\n"
              f"  Avg Total:  {epoch_loss_total / num_batches:.5f}\n"
              f"  Avg Recon:  {epoch_loss_recon / num_batches:.5f}\n"
              f"  Avg Ref:    {epoch_loss_ref / num_batches:.5f}\n"
              f"  Avg Smooth: {epoch_loss_smooth / num_batches:.5f}\n"
              f"  Avg Illum:  {epoch_loss_illum / num_batches:.5f}\n"
              f"  LR:         {optimizer.param_groups[0]['lr']:.7f}\n")

        # --- 每个 Epoch 保存一次 ---
        utils.logging.save_checkpoint(
            {'epoch': epoch, 'step': global_step,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()},
            filename=os.path.join(config.data.ckpt_dir, f'epoch_{epoch}')
        )

    print("\n" + "=" * 60)
    print("Stage 1 训练完成!")
    print(f"最终模型保存在: {config.data.ckpt_dir}/stage1_weight.pth.tar")
    print("可以直接用于 Stage 2 的 train.py")
    print("=" * 60)


# ===================== 验证函数 =====================
def validate_stage1(model, val_loader, device, args, step):
    """
    验证: 对验证集图像做分解并保存结果
    保存内容: 原图、反射图R、光照图L、重建图 R*L
    """
    save_dir = os.path.join(args.image_folder, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)

    for i, (x, img_ids) in enumerate(val_loader):
        x = x.to(device)
        low_img = x[:, :3, :, :]
        _, _, h, w = low_img.shape

        output = model(x, pred_fea=None)

        low_R = output["low_R"]        # [B, 3, H/8, W/8]
        low_L = output["low_L"]        # [B, 3, H/8, W/8]
        low_fea = output["low_fea"]    # [B, 3, H/8, W/8] 特征图(CTDN重建目标)
        recon_fea = low_R * low_L      # 重建的特征图

        # 上采样回原图尺寸仅用于可视化
        low_R_up = F.interpolate(low_R, size=(h, w), mode='bilinear', align_corners=False)
        low_L_up = F.interpolate(low_L, size=(h, w), mode='bilinear', align_corners=False)
        low_fea_up = F.interpolate(low_fea, size=(h, w), mode='bilinear', align_corners=False)
        recon_fea_up = F.interpolate(recon_fea, size=(h, w), mode='bilinear', align_corners=False)

        # 只保存前几个样本
        if i >= 2:
            break

        for j in range(min(x.shape[0], 2)):
            img_id = img_ids[j] if isinstance(img_ids, list) else f"{i}_{j}"
            utils.logging.save_image(low_img[j:j+1], os.path.join(save_dir, f"{img_id}_low_input.png"))
            utils.logging.save_image(low_fea_up[j:j+1], os.path.join(save_dir, f"{img_id}_fea.png"))
            utils.logging.save_image(low_R_up[j:j+1], os.path.join(save_dir, f"{img_id}_R.png"))
            utils.logging.save_image(low_L_up[j:j+1].mean(dim=1, keepdim=True).repeat(1,3,1,1),
                                     os.path.join(save_dir, f"{img_id}_L.png"))
            utils.logging.save_image(recon_fea_up[j:j+1], os.path.join(save_dir, f"{img_id}_recon_fea.png"))

    print(f"  验证图像已保存到 {save_dir}")


if __name__ == "__main__":
    main()
