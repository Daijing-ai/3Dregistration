import argparse
import cv2
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def depth_to_pointcloud(depth_map, K, rgb_image=None):
    """
    将深度图转换为点云
    :param depth_map: 深度图 (H x W)
    :param K: 相机内参矩阵 (3x3)
    :param rgb_image: 对应的RGB图像 (H x W x 3) - 可选，用于着色
    :return: open3d.PointCloud对象
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 创建齐次坐标
    uv_homogeneous = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    
    # 转换到相机坐标系
    points_cam = (np.linalg.inv(K) @ uv_homogeneous.T).T
    points_cam *= depth_map.flatten()[:, None]
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam)
    
    # 添加颜色（如果提供了RGB图像）
    if rgb_image is not None:
        colors = rgb_image.reshape(-1, 3) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def preprocess_pointcloud(pcd, voxel_size=2.0):
    """
    点云预处理：降采样、去噪
    :param pcd: 输入点云
    :param voxel_size: 体素降采样尺寸
    :return: 处理后的点云
    """
    # 体素降采样
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 统计离群点去除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 半径离群点去除
    cl, ind = cl.remove_radius_outlier(nb_points=16, radius=5.0)
    
    # 估计法线（用于后续配准）
    cl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    
    return cl




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default='/home/guestdj/nvme3n1/DJ/vggt/clip_keyframes',type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = args.encoder # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder]).to(DEVICE)
    depth_anything.load_state_dict(torch.load(f'/home/guestdj/nvme3n1/DJ/summer_school_project/depth_anything_vitb14.pth'))

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    # 相机内参 - 需要根据您的腹腔镜进行标定
    # 这是一个典型值，请替换为您的实际相机参数
    K = np.array([
        [4578, 0, 960],  # fx, 0, cx    
        [0, 4578, 540],   # 0, fy, cy    
        [0, 0, 1]         # 0, 0, 1
    ])

    outdir = args.outdir
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_maps"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "pointclouds"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "meshes"), exist_ok=True)  # 新增mesh输出目录

    pointclouds = []
    for filename in tqdm(filenames):
        
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image_tensor = transform({'image': image})['image']
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image_tensor)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_vis = (depth_norm * 255.0).cpu().numpy().astype(np.uint8)
        depth_np = depth.cpu().numpy()

        # 保存深度图和彩色可视化
        if args.grayscale:
            depth_color = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # 拼接原图和深度图
        raw_bgr = raw_image
        split_region = np.ones((raw_bgr.shape[0], 20, 3), dtype=np.uint8) * 255
        paired = cv2.hconcat([raw_bgr, split_region, depth_color])

        base = os.path.splitext(os.path.basename(filename))[0]
        cv2.imwrite(os.path.join(outdir, "depth_maps", f"{base}_pair.png"), paired)
        cv2.imwrite(os.path.join(outdir, "depth_maps", f"{base}_depth.png"), depth_color)

        # 生成点云
        pcd = depth_to_pointcloud(depth_np, K, raw_bgr)
        pcd = preprocess_pointcloud(pcd, voxel_size=2.0)
        ply_path = os.path.join(outdir, "pointclouds", f"{base}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)

        # 收集每帧点云用于后续融合
        pointclouds.append(pcd)

    # 融合所有点云并保存
    if pointclouds:
        merged_pcd = pointclouds[0]
        for pcd in pointclouds[1:]:
            merged_pcd += pcd

        # 再次降采样和去噪
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=2.0)
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 保存融合后的点云
        o3d.io.write_point_cloud(
            os.path.join(outdir, "merged_pointcloud.ply"),
            merged_pcd
        )
        print(f"Saved merged point cloud with {len(merged_pcd.points)} points")