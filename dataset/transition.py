import os
import argparse

def generate_single_dataset_txt(data_dir, output_txt_name="dataset.txt"):
    low_img_dir = os.path.join(data_dir, "input")
    high_img_dir = os.path.join(data_dir, "target")
    
    # 检查目录有效性
    if not os.path.isdir(low_img_dir) or not os.path.isdir(high_img_dir):
        raise FileNotFoundError(f"低光目录 {low_img_dir} 或正常光目录 {high_img_dir} 不存在")
    
    # 获取所有图像文件（支持常见格式）
    img_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    low_imgs = sorted([f for f in os.listdir(low_img_dir) if f.lower().endswith(img_formats)])
    high_imgs = sorted([f for f in os.listdir(high_img_dir) if f.lower().endswith(img_formats)])
    
    # 验证图像数量匹配
    if len(low_imgs) != len(high_imgs):
        raise ValueError(f"低光图像数量（{len(low_imgs)}）与正常光图像数量（{len(high_imgs)}）不一致")
    
    # 生成TXT文件
    output_path = os.path.join(data_dir, output_txt_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        for low_img, high_img in zip(low_imgs, high_imgs):
            # 获取绝对路径（也可改为相对路径，去掉os.path.abspath即可）
            low_path = os.path.abspath(os.path.join(low_img_dir, low_img))
            high_path = os.path.abspath(os.path.join(high_img_dir, high_img))
            f.write(f"{low_path} {high_path}\n")
    
    print(f"TXT文件生成完成！")
    print(f"文件路径：{output_path}")
    print(f"包含数据条数：{len(low_imgs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成单一数据集TXT文件")
    parser.add_argument("--data_dir",  required=True, help="数据集根目录（需包含low和high子目录）")
    parser.add_argument("--output_txt", default="dataset.txt", help="输出TXT文件名（默认dataset.txt）")
    args = parser.parse_args()
    
    generate_single_dataset_txt(
        data_dir=args.data_dir,
        output_txt_name=args.output_txt
    )
