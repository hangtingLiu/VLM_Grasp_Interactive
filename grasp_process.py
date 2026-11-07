import os
import sys
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spatialmath as sm

from manipulator_grasp.arm.motion_planning import *

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net




# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 5000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d



# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, sam_mask_path=None, *, visual=True, all_candidates=False):
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据，此处使用返回的工作空间掩码路径
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path)

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    if not len(gg):
        return gg

    # 5. 碰撞检测
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        if all_candidates:
            height, width = depth_path.shape
            focal = height / (2 * np.tan(np.pi / 8))
            camera = CameraInfo(width, height, focal, focal, width / 2, height / 2, 1.)
            scene = create_point_cloud_from_depth_image(depth_path, camera, organized=True)
            valid = np.isfinite(depth_path) & (depth_path > 0) & (depth_path < 2)
            points = scene[valid]
            gg = gg.nms().sort_by_score()
        else:
            points = np.asarray(cloud_o3d.points)
        mfcdetector = ModelFreeCollisionDetector(points, voxel_size=0.005 if all_candidates else voxel_size)
        if all_candidates:
            collision_mask, empty = mfcdetector.detect(
                gg, approach_dist=0.05, collision_thresh=collision_thresh, return_empty_grasp=True)
            collision_mask |= empty
            if collision_mask.all():
                # Thin objects can be rejected by the visible-point heuristic.
                # The caller must still validate contact and placement in MuJoCo.
                print("严格点云筛选无候选，改用较宽松的点云阈值后进行仿真验证。", flush=True)
                collision_mask = mfcdetector.detect(
                    gg, approach_dist=0.05, collision_thresh=0.05)
        else:
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]
    if all_candidates or not len(gg):
        return gg

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()

    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第一列
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # # ===== 新增：利用 SAM 生成的目标掩码过滤抓取预测（投影到图像坐标判断） =====
    # if sam_mask_path is not None:
    #     # 加载 SAM 目标掩码
    #     if isinstance(sam_mask_path, str):
    #         sam_mask = np.array(Image.open(sam_mask_path))
    #     elif isinstance(sam_mask_path, np.ndarray):
    #         sam_mask = sam_mask_path
    #     else:
    #         raise TypeError("sam_mask_path 既不是字符串路径也不是 NumPy 数组！")
    #     # 假定 SAM 掩码与颜色图尺寸一致（640x640）
    #     height, width = sam_mask.shape[:2]
    #     # 动态计算相机内参（基于垂直视场角 fovy）
    #     fovy = np.pi / 4  # 垂直视场角，假设为 45 度
    #     focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（像素单位）
    #     cx = width / 2.0   # 光心 X 坐标（图像中心）
    #     cy = height / 2.0  # 光心 Y 坐标（图像中心）

    #     sam_filtered = []
    #     for grasp in filtered:
    #         # grasp.translation 为摄像头坐标系下的 3D 坐标 [X, Y, Z]
    #         X, Y, Z = grasp.translation
    #         if Z <= 0:
    #             continue
    #         u = focal * X / Z + cx
    #         v = focal * Y / Z + cy
    #         u_int = int(round(u))
    #         v_int = int(round(v))
    #         # 检查投影点是否在图像范围内（640x640）
    #         if u_int < 0 or u_int >= 640 or v_int < 0 or v_int >= 640:
    #             continue
    #         # 若 SAM 掩码中该像素有效（非0），则保留
    #         if sam_mask[v_int, u_int] > 0:
    #             sam_filtered.append(grasp)
    #     if len(sam_filtered) == 0:
    #         print("\n[Warning] No grasp predictions fall inside the SAM mask. Using previous predictions.")
    #     else:
    #         print(f"\n[DEBUG] Filtered {len(sam_filtered)} grasps inside the SAM mask out of {len(filtered)} predictions.")
    #         filtered = sam_filtered

    # ===== 新增部分：计算物体中心点 =====
    # 使用点云计算物体的中心点
    points = np.asarray(cloud_o3d.points)
    object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    # 计算每个抓取位姿中心点与物体中心点的距离
    distances = []
    for grasp in filtered:
        grasp_center = grasp.translation
        distance = np.linalg.norm(grasp_center - object_center)
        distances.append(distance)

    # 创建一个新的排序列表，包含距离和抓取对象
    grasp_with_distances = [(g, d) for g, d in zip(filtered, distances)]
    
    # 按距离升序排序（距离越小越好）
    grasp_with_distances.sort(key=lambda x: x[1])
    
    # 提取排序后的抓取列表
    filtered = [g for g, d in grasp_with_distances]

    # ===== 新增部分：综合得分和距离进行最终排序 =====
    # 创建一个新的排序列表，包含综合得分和抓取对象
    # 综合得分 = 抓取得分 * 0.7 + (1 - 距离/最大距离) * 0.3
    max_distance = max(distances) if distances else 1.0
    grasp_with_composite_scores = []

    for g, d in grasp_with_distances:
        # 归一化距离分数（距离越小分数越高）
        distance_score = 1 - (d / max_distance)
        
        # 综合得分 = 抓取得分 * 权重1 + 距离得分 * 权重2
        composite_score = g.score * 0.1 + distance_score * 0.9
        # print(f"\n g.score = {g.score}, distance_score = {distance_score}")
        grasp_with_composite_scores.append((g, composite_score))

    # 按综合得分降序排序
    grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的抓取列表
    filtered = [g for g, score in grasp_with_composite_scores]


    # # 对过滤后的抓取根据 score 排序（降序）
    # filtered.sort(key=lambda g: g.score, reverse=True)

    # 取第1个抓取
    top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]

    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取

    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    return new_gg

    #return best_translation, best_rotation, best_width



# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg, *, direct_approach=False, after_lift=None):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    # n_wc = np.array([0.0, -1.0, 0.0]) # 相机朝向
    # o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向 [0.5, 0.0, -1.0] -> [-1.0, 0.0, -0.5]
    # t_wc = np.array([1.0, 0.6, 2.0]) # 相机的位置。2.0是相机高度，与scene.xml中保持一致。
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.85, 0.8, 1.6]) 

    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    time1 = 1
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    if direct_approach:
        robot.set_joint(q1)
        q1 = robot.ikine(T2)
        robot.set_joint(q0)
        if len(q1) != 6 or not np.isfinite(q1).all():
            raise ValueError("接近点无有效逆解")
        q1 = q0 + (q1 - q0 + np.pi) % (2 * np.pi) - np.pi
        time1 = 2.5
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    # 执行planner_array = [planner1]
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 2.接近抓取位姿
    # 目标：从预抓取位姿直线移动到抓取点附近（T2）
    # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
    if not direct_approach:
        _move_to_approach(env, robot, action, q1, T2)

    _finish_grasp(env, robot, action, q0, T2, T_wo, after_lift)


def _move_to_approach(env, robot, action, q1, T2):
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
    velocity_parameter1 = QuinticVelocityParameter(time2) # 速度曲线（五次多项式插值）
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    # 执行planner_array = [planner2]
    time_array = [0.0, time2]
    planner_array = [planner2]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

def _finish_grasp(env, robot, action, q0, T2, T_wo, after_lift):
    # 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。通过逐步增加 action[-1]（夹爪控制信号）闭合夹爪，抓取物体。
    time3 = 1
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    # 执行planner_array = [planner3]
    time_array = [0.0, time3]
    planner_array = [planner3]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num) 
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)): 
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)

    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5 # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # 执行planner_array = [planner4, planner5, planner6]
    time_array = [0.0, time4, time5, time6]
    planner_array = [planner4, planner5, planner6]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    lift_checked = False
    for timei in times:
        if after_lift is not None and not lift_checked and timei > time4:
            after_lift()
            lift_checked = True
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
