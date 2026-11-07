import os.path
import sys

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

import glfw
import cv2

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj


class UR5GraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        self.fovy = np.pi / 4

        # 新增离屏渲染相关属性
        self.camera_name = "cam"
        self.camera_id = -1
        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_viewport = None
        self.glfw_window = None


    def reset(self, *, graphics=True):

        # 初始化 MuJoCo 模型和数据
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 创建机械臂实例并设置其基座位置
        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        # 设置机械臂的初始关节角度，并同步到 MuJoCo 模型
        self.robot_q = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, -np.pi / 2 * 0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # 添加约束，将自由关节的姿态固定为机械臂末端执行器的姿态
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q)) # 将自由关节的姿态固定为机械臂末端执行器的姿态
        # 定义机械臂末端执行器的工具偏移
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        # 计算机械臂末端执行器的初始姿态。
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        if not graphics:
            return

        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 启用深度渲染
        self.mj_depth_renderer.enable_depth_rendering()
        
        # 初始化被动查看器
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        # 为了方便观察
        self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
        self.mj_viewer.cam.azimuth = 210      # 对应XML中的azimuth
        self.mj_viewer.cam.elevation = -35    # 对应XML中的elevation
        self.mj_viewer.cam.distance = 1.2     # 根据场景调整的距离值
        self.mj_viewer.sync() # 立即同步更新

        # # --- 新增: 初始化离屏渲染 ---
        # # 初始化GLFW用于离屏渲染
        # glfw.init()
        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        # self.glfw_window = glfw.create_window(self.width, self.height, "Offscreen", None, None)
        # glfw.make_context_current(self.glfw_window)

        # # 获取相机ID
        # self.camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # if self.camera_id != -1:
        #     print(f"成功找到相机 '{self.camera_name}', ID: {self.camera_id}")
        #     # 使用XML中定义的固定相机
        #     self.offscreen_camera = mujoco.MjvCamera()
        #     mujoco.mjv_defaultCamera(self.offscreen_camera)
        #     self.offscreen_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        #     self.offscreen_camera.fixedcamid = self.camera_id

        # # 创建离屏场景和上下文
        # self.offscreen_scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
        # self.offscreen_context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # self.offscreen_viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.offscreen_context)

        # # 创建OpenCV窗口
        # cv2.namedWindow('MuJoCo Camera Output', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('MuJoCo Camera Output', self.width, self.height)


    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

        # 清理离屏渲染资源
        cv2.destroyAllWindows()
        if self.glfw_window is not None:
            glfw.destroy_window(self.glfw_window)
        glfw.terminate()
        self.offscreen_context = None
        self.offscreen_scene = None

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None:
            self.mj_viewer.sync()

        # # --- 新增: 离屏渲染和显示 ---
        # if all([self.offscreen_context, self.offscreen_scene, self.offscreen_camera]):
        #     # 更新场景
        #     mujoco.mjv_updateScene(self.mj_model, self.mj_data, mujoco.MjvOption(), 
        #                          mujoco.MjvPerturb(), self.offscreen_camera, 
        #                          mujoco.mjtCatBit.mjCAT_ALL.value, self.offscreen_scene)
            
        #     # 渲染到离屏缓冲区
        #     mujoco.mjr_render(self.offscreen_viewport, self.offscreen_scene, self.offscreen_context)
            
        #     # 读取像素数据
        #     rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        #     mujoco.mjr_readPixels(rgb, None, self.offscreen_viewport, self.offscreen_context)
            
        #     # 转换颜色空间并显示
        #     bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        #     cv2.imshow('MuJoCo Camera Output', bgr)
            
        #     # 检查ESC键
        #     if cv2.waitKey(1) == 27:
        #         print("用户按下了ESC键,退出仿真。")
        #         self.close()
        #         exit(0)
                
    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 渲染图像和深度图
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }



if __name__ == '__main__':
    env = UR5GraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
