import cv2
import numpy as np
import os
import glob

# 设置路径
data_dir = '/home/guestdj/nvme3n1/DJ/vggt/clip_keyframes'
mask_results_dir = '/home/guestdj/nvme3n1/DJ/summer_school_project/mask_results'
# 确保掩膜结果目录存在
os.makedirs(mask_results_dir, exist_ok=True)

# 获取所有jpg图片
jpg_files = glob.glob(os.path.join(data_dir, '*.jpg'))

for jpg_file in jpg_files:
    # 处理文件名
    base_name = os.path.splitext(os.path.basename(jpg_file))[0]
    base_name = base_name.split('_')[0] + "_" + base_name.split('_')[1]
    
    # 构建路径
    rgb_path = jpg_file
    mask_path = os.path.join(mask_results_dir, base_name + '.png')
    output_dir = os.path.join(mask_results_dir, base_name + '_extracted')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像和掩膜
    rgb_img = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查文件是否成功读取
    if rgb_img is None:
        print(f"RGB图像未找到: {rgb_path}")
        continue
    if mask is None:
        print(f"掩膜图像未找到: {mask_path}")
        continue
    
    # 调整掩膜尺寸
    if rgb_img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # 处理每个唯一值
    unique_values = np.unique(mask)
    print(f"{rgb_path}: {unique_values}")
    for value in unique_values:                                             
        if value == 0:  # 跳过背景
            continue
            
        # 创建二值掩膜
        value_mask = (mask == value).astype(np.uint8) * 255
        
        # 提取区域
        result = np.zeros_like(rgb_img)
        result[value_mask > 0] = rgb_img[value_mask > 0]
        
        # 保存结果
        mask_output = os.path.join(output_dir, f"{base_name}_mask_value_{value}.png")
        region_output = os.path.join(output_dir, f"{base_name}_region_value_{value}.png")
        
        cv2.imwrite(mask_output, value_mask)
        cv2.imwrite(region_output, result)
    
    print(f"处理完成: {base_name}，提取了{len(unique_values)-1}个区域")

"""
nohup nnUNetv2_predict -i [待预测文件夹] -o [目标文件夹] -d 007 -c 2d -f 0 1 2 3 4 > predict.log 2>&1 &
"""