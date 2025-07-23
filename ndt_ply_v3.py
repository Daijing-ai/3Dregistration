import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    """预处理点云：下采样和法线估计"""
    # 下采样点云
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 估计法线
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down

def execute_registration(source, target, voxel_size, init_trans=np.eye(4)):
    """执行点云配准，自动选择最佳方法"""
    # 预处理点云
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    # 尝试使用NDT方法（如果可用）
    try:
        # 新版本Open3D的NDT实现
        ndt = o3d.pipelines.registration.TransformationEstimationForNDT(
            voxel_size=voxel_size,
            max_iterations=100  # 增加迭代次数
        )
        
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=100
        )
        
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, 
            max_correspondence_distance=voxel_size * 3,
            init=init_trans,
            estimation_method=ndt,
            criteria=criteria
        )
        return result
    except AttributeError:
        # 旧版本Open3D的NDT实现
        try:
            result = o3d.pipelines.registration.registration_ndt(
                source_down, target_down, 
                voxel_size=voxel_size,
                max_iterations=100,
                init=o3d.core.Tensor(init_trans)
            )
            return result
        except AttributeError:
            # 如果NDT不可用，使用Point-to-Plane ICP作为备选
            print("NDT不可用，使用Point-to-Plane ICP作为备选方案")
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=100
            )
            
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, 
                max_correspondence_distance=voxel_size * 3,
                init=init_trans,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=criteria
            )
            return result

def prepare_dataset(source, target, voxel_size):
    """准备配准数据集（粗配准）"""
    # 下采样
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # 估计法线
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size*2, max_nn=50))  # 增加邻域点数量
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size*2, max_nn=50))
    
    # 计算FPFH特征
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=150))  # 增加邻域点
    
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=150))
    
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """执行全局粗配准"""
    distance_threshold = voxel_size * 4.5  # 优化距离阈值
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))  # 增加迭代次数
    return result

# 主程序
if __name__ == "__main__":
    # 读取点云
    source_path = "/home/guestdj/nvme3n1/DJ/Depth-Anything/ply/UMD_1_t2_cor.nii.gz.ply"
    target_path = "/home/guestdj/nvme3n1/DJ/Depth-Anything/ply/glbscene_50_All_maskbFalse_maskwFalse_camFalse_skyFalse_predPointmap_Branch(1).ply"
    
    print(f"加载源点云: {source_path}")
    source = o3d.io.read_point_cloud(source_path)
    
    print(f"加载目标点云: {target_path}")
    target = o3d.io.read_point_cloud(target_path)
    
    # 检查点云是否为空
    if not source.has_points() or len(source.points) == 0:
        raise ValueError("源点云读取失败或为空！")
    if not target.has_points() or len(target.points) == 0:
        raise ValueError("目标点云读取失败或为空！")
    
    print(f"源点云点数: {len(source.points)}")
    print(f"目标点云点数: {len(target.points)}")
    
    # 打印Open3D版本信息
    print(f"使用的Open3D版本: {o3d.__version__}")
    
    # 自动确定初始体素大小
    bbox = source.get_axis_aligned_bounding_box()
    size = np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound())
    VOXEL_SIZE = 0.8  # 根据点云尺寸自动设置
    
    print(f"自动确定的初始体素大小: {VOXEL_SIZE:.4f}")
    
    # 执行全局粗配准
    print("执行全局粗配准...")
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, VOXEL_SIZE)
    
    print(f"粗配准输入点云: 源={len(source_down.points)}, 目标={len(target_down.points)}")
    
    global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, VOXEL_SIZE)
    print(f"粗配准结果 fitness: {global_result.fitness}")
    
    # 粗配准失败处理策略
    if global_result.fitness < 0.3:
        print("粗配准质量较低，尝试优化...")
        
        # 策略1: 增大体素尺寸
        original_voxel = VOXEL_SIZE
        VOXEL_SIZE *= 1.5
        print(f"尝试增大体素尺寸至: {VOXEL_SIZE:.4f}")
        source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, VOXEL_SIZE)
        global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, VOXEL_SIZE)
        print(f"优化后的粗配准结果 fitness: {global_result.fitness}")
        
        # 策略2: 如果仍然失败，尝试初始变换
        if global_result.fitness < 0.3:
            print("尝试使用初始变换策略...")
            # 恢复原始体素大小
            VOXEL_SIZE = original_voxel
            
            # 使用基于PCA的初始对齐
            source_points = np.asarray(source.points)
            target_points = np.asarray(target.points)
            
            # 计算重心
            source_center = np.mean(source_points, axis=0)
            target_center = np.mean(target_points, axis=0)
            
            # 计算协方差矩阵
            source_cov = np.cov(source_points.T)
            target_cov = np.cov(target_points.T)
            
            # 计算特征向量
            _, source_evecs = np.linalg.eigh(source_cov)
            _, target_evecs = np.linalg.eigh(target_cov)
            
            # 构建初始变换矩阵
            init_trans = np.eye(4)
            init_trans[:3, :3] = target_evecs @ np.linalg.inv(source_evecs)
            init_trans[:3, 3] = target_center - source_center
            
            global_result.transformation = init_trans
            print("使用PCA初始变换矩阵")
    
    # 执行精配准
    print("执行精配准...")
    result = execute_registration(source, target, VOXEL_SIZE, init_trans=global_result.transformation)

    # 输出配准结果
    print("\n配准结果:")
    print(f"变换矩阵:\n{np.round(result.transformation, 4)}")
    print(f"匹配度: {result.fitness:.4f} (大于0.5表示良好)")
    print(f"均方误差: {result.inlier_rmse:.6f}")
    
    # 应用变换到源点云
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(result.transformation)
    
    # 保存配准后的点云
    output_path = "registered_result.ply"
    o3d.io.write_point_cloud(output_path, source_transformed)
    print(f"配准后的点云已保存为 '{output_path}'")
    
    # 保存变换矩阵到文本文件
    matrix_path = "transformation_matrix.txt"
    np.savetxt(matrix_path, result.transformation, fmt='%.6f')
    print(f"变换矩阵已保存为 '{matrix_path}'")
    
    # 计算配准误差统计
    dists = source_transformed.compute_point_cloud_distance(target)
    dists = np.asarray(dists)
    mean_error = np.mean(dists)
    max_error = np.max(dists)
    min_error = np.min(dists)
    
    print("\n配准误差统计:")
    print(f"平均误差: {mean_error:.6f} 米")
    print(f"最大误差: {max_error:.6f} 米")
    print(f"最小误差: {min_error:.6f} 米")
    
    # 保存误差统计到文件
    stats_path = "registration_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"变换矩阵:\n{np.round(result.transformation, 4)}\n\n")
        f.write(f"匹配度: {result.fitness:.4f}\n")
        f.write(f"均方误差: {result.inlier_rmse:.6f}\n")
        f.write(f"平均误差: {mean_error:.6f}\n")
        f.write(f"最大误差: {max_error:.6f}\n")
        f.write(f"最小误差: {min_error:.6f}\n")
    
    print(f"配准统计信息已保存为 '{stats_path}'")
    print("配准流程完成！")