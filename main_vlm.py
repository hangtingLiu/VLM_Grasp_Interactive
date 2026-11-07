"""Run one text-directed pick and place in MuJoCo."""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "manipulator_grasp"))

from grasp_process import execute_grasp, run_grasp_inference
from graspnetAPI import GraspGroup
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from vlm_process import segment_image, voice_command_to_keyword

MAX_TRIALS = 12
SEARCH_SECONDS = 20


def scene_groups(model):
    robots = {model.body(name).id for name in ("ur5e_base", "2f85_base")}
    objects = set(model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]) - robots
    groups = np.arange(model.nbody)
    roots = robots | objects
    for body in range(1, model.nbody):
        parent = groups[model.body_parentid[body]]
        if body not in roots and parent in roots:
            groups[body] = parent
    return objects, robots, groups[model.geom_bodyid]


class UnexpectedContact(RuntimeError):
    pass


class SearchTimeout(RuntimeError):
    pass


class EmptyGrasp(RuntimeError):
    pass


class GraspTrial(UR5GraspEnv):
    deadline = None

    def watch_contacts(self, target):
        model = self.mj_model
        objects, self.robot_groups, self.geometry_groups = scene_groups(model)
        self.obstacles = objects - {target}
        self.table_groups = {model.body(name).id for name in ("simple_table", "table1", "table2")}
        self.moving_groups = self.robot_groups | {target}
        self.target_body = self.mj_data.body(target)
        self.peak_height = float(self.target_body.xpos[2])

    def step(self, action=None):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchTimeout
        if self.mj_viewer is not None and not self.mj_viewer.is_running():
            raise KeyboardInterrupt
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        for contact in self.mj_data.contact:
            if contact.dist > 0 or min(contact.geom) < 0:
                continue
            a, b = self.geometry_groups[contact.geom]
            object_contact = ((a in self.moving_groups and b in self.obstacles) or
                              (b in self.moving_groups and a in self.obstacles))
            table_contact = ((a in self.robot_groups and b in self.table_groups) or
                             (b in self.robot_groups and a in self.table_groups))
            if object_contact or table_contact:
                raise UnexpectedContact(
                    f"{self.mj_model.body(a).name} / {self.mj_model.body(b).name}")
        self.peak_height = max(self.peak_height, self.target_body.xpos[2])
        if self.mj_viewer is not None:
            self.mj_viewer.sync()


def top_down_candidates(depth, mask, model, data):
    """Sample narrow cross-sections of the masked object in tabletop coordinates."""
    y, x = np.nonzero((mask > 0) & np.isfinite(depth) & (depth > 0) & (depth < 2))
    if len(x) < 30:
        return np.empty((0, 17))
    height, width = depth.shape
    camera = model.camera("cam").id
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
    z = depth[y, x]
    optical = np.column_stack(((x - width / 2) * z / focal, (y - height / 2) * z / focal, z))
    rotation = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    origin = data.cam_xpos[camera]
    world = optical @ rotation.T + origin
    table = model.geom("simple_table").id
    tabletop = data.geom_xpos[table, 2] + model.geom_size[table, 2]
    world = world[world[:, 2] > tabletop + 0.002]
    if len(world) < 30:
        return np.empty((0, 17))
    # Estimate clearance from other visible objects, excluding the tabletop.
    oy, ox = np.nonzero((mask == 0) & np.isfinite(depth) & (depth > 0) & (depth < 2))
    oz = depth[oy, ox]
    obstacles = np.column_stack(((ox - width / 2) * oz / focal,
                                 (oy - height / 2) * oz / focal, oz))
    obstacles = obstacles @ rotation.T + origin
    obstacles = obstacles[obstacles[:, 2] > tabletop + 0.008, :2]
    _, _, vectors = np.linalg.svd(world[:, :2] - world[:, :2].mean(axis=0), full_matrices=False)
    along, across = vectors
    longitudinal = world[:, :2] @ along
    lateral = world[:, :2] @ across
    sections = []
    for fraction in (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92):
        station = np.quantile(longitudinal, fraction)
        local = np.abs(longitudinal - station) < 0.009
        if np.count_nonzero(local) < 15:
            continue
        left, right = np.quantile(lateral[local], [0.05, 0.95])
        span = right - left
        if not 0.002 < span < 0.085:
            continue
        xy = station * along + ((left + right) / 2) * across
        surface = np.quantile(world[local, 2], 0.75)
        clearance = np.min(np.linalg.norm(obstacles - xy, axis=1), initial=np.inf)
        sections.append((fraction, clearance, span, xy, surface))

    approach = np.array([0., 0., -1.])
    rows = []
    # Cover both ends early on elongated objects before spending trials in the middle.
    elongated = len(world) > 0 and np.ptp(longitudinal) > 2 * np.ptp(lateral)
    sections.sort(key=lambda section: (
        elongated and 0.08 < section[0] < 0.92,
        2 * section[2] - min(section[1], 0.06),
    ))
    for direction in (1, -1):
        for clearance in (0.045, 0.055, 0.035, 0.025):
            for _, _, span, xy, surface in sections:
                position = np.r_[xy, max(surface + clearance, tabletop + 0.047)]
                closing = np.r_[direction * across, 0.]
                orientation = np.column_stack((approach, closing, np.cross(approach, closing)))
                rows.append(np.r_[0., span, 0.02, 0.02, (rotation.T @ orientation).ravel(),
                                  rotation.T @ (position - origin), -1.])
    return np.asarray(rows).reshape(-1, 17)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", help="例如：抓苹果")
    parser.add_argument("--voice", action="store_true", help="通过默认麦克风输入中文指令")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--verify-body", help="可选，要求识别结果与指定刚体一致")
    args = parser.parse_args()
    if args.voice and args.command is not None:
        parser.error("--voice 与文字指令只能选一个")
    if args.headless and args.command is None and not args.voice:
        parser.error("--headless 需要文字指令或 --voice")
    os.chdir(ROOT)
    command = voice_command_to_keyword() if args.voice else args.command
    if args.voice and not command:
        raise SystemExit("没有识别到语音指令，请检查麦克风后重试。")
    if args.voice:
        confirmed = input(f"识别结果「{command}」。按回车执行，或输入正确指令（输入 q 退出）：").strip()
        if confirmed.lower() == "q":
            return
        command = confirmed or command
    out = ROOT / "outputs" / "main_vlm"
    out.mkdir(parents=True, exist_ok=True)
    env = GraspTrial()
    try:
        env.reset(graphics=not args.headless)
        model, data = env.mj_model, env.mj_data
        mujoco.mj_step(model, data, nstep=500)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=env.height, width=env.width) as renderer:
            renderer.update_scene(data, camera="cam")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            object_ids = renderer.render().copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / "rgb.png"), bgr)
        mask = segment_image(bgr, str(out / "mask.png"), command=command,
                             speak=args.voice, interactive=not args.headless)
        if mask is None or mask.shape != depth.shape:
            raise SystemExit("未生成有效目标掩码。")
        depth[~np.isfinite(depth) | (depth <= 0)] = np.inf
        if not np.any((mask > 0) & (depth < 2)):
            raise SystemExit("目标掩码内没有有效深度。")
        pixels = object_ids[mask > 0]
        geoms = pixels[pixels[:, 1] == mujoco.mjtObj.mjOBJ_GEOM, 0]
        objects, _, geometry_groups = scene_groups(model)
        bodies = geometry_groups[geoms]
        counts = {body: int(np.count_nonzero(bodies == body)) for body in objects}
        target_id = max(counts, key=counts.get, default=None)
        if target_id is None or not counts[target_id]:
            raise SystemExit("分割掩码未覆盖可抓取物体。")
        target = model.body(target_id).name
        if args.verify_body and args.verify_body != target:
            raise SystemExit(
                f"识别到 {target}，但 --verify-body 要求 {args.verify_body}。"
                f"请改用 --verify-body {target}，或省略该参数。")
        grasps = run_grasp_inference(rgb, depth, mask, visual=False, all_candidates=True)
        print(f"点云过滤后保留 {len(grasps)} 个候选。", flush=True)
        # Try the highest-ranked learned poses first, then geometry-based poses.
        learned = grasps[:min(4, len(grasps))]
        geometric = top_down_candidates(depth, mask, model, data)
        fallback_end = len(learned) + len(geometric)
        candidates = GraspGroup(np.vstack((learned.grasp_group_array, geometric,
                                           grasps[len(learned):].grasp_group_array)))
        print(f"另生成 {len(geometric)} 个基于形状的俯抓候选。", flush=True)
        if not len(candidates):
            raise SystemExit("没有可执行的抓取候选。")
        env.watch_contacts(target_id)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial_time = data.time
        initial = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, initial, spec)
        start_joint = env.robot.get_joint().copy()
        initial_height = env.peak_height

        def restore():
            mujoco.mj_setState(model, data, initial, spec)
            mujoco.mj_forward(model, data)
            env.robot.set_joint(start_joint)
            env.peak_height = initial_height

        viewer = env.mj_viewer
        # Simulate each trial privately; only replay a passing trajectory in the viewer.
        env.mj_viewer = None
        started = time.monotonic()
        env.deadline = started + SEARCH_SECONDS

        def verify_lift():
            if env.target_body.xpos[2] - initial_height < 0.08:
                raise EmptyGrasp

        try:
            for index in range(min(MAX_TRIALS, len(candidates))):
                restore()
                try:
                    execute_grasp(env, candidates[index:index + 1],
                                  direct_approach=len(learned) <= index < fallback_end,
                                  after_lift=verify_lift)
                    for _ in range(500):
                        env.step()
                except UnexpectedContact as exc:
                    print(f"候选 {index + 1} 已跳过，仿真 {data.time - initial_time:.2f} 秒时接触 {exc}。", flush=True)
                    continue
                except EmptyGrasp:
                    print(f"候选 {index + 1} 未夹稳，跳过放置。", flush=True)
                    continue
                except SearchTimeout:
                    restore()
                    raise SystemExit(f"后台抓取搜索达到 {SEARCH_SECONDS} 秒，未找到可用姿态。")
                except (AssertionError, ValueError) as exc:
                    print(f"候选 {index + 1} 运动规划未通过：{exc or '逆解不可用'}。", flush=True)
                    continue
                mujoco.mj_forward(model, data)
                final = env.target_body.xpos.copy()
                lift = env.peak_height - initial_height
                distance = np.linalg.norm(final[:2] - [1.4, 0.3])
                print(f"候选 {index + 1}，抬升 {lift:.3f} 米，放置位置 {final}，"
                      f"距目标 {distance:.3f} 米", flush=True)
                if lift >= 0.1 and distance < 0.12 and abs(final[2] - initial_height) < 0.05:
                    print(f"抓取与放置通过，未检测到意外接触。后台搜索耗时 "
                          f"{time.monotonic() - started:.1f} 秒。", flush=True)
                    break
            else:
                restore()
                raise SystemExit(f"已检查 {min(MAX_TRIALS, len(candidates))} 个候选，未找到可用姿态。")
        finally:
            env.deadline = None
            env.mj_viewer = viewer
        if viewer is not None:
            restore()
            execute_grasp(env, candidates[index:index + 1],
                          direct_approach=len(learned) <= index < fallback_end)
            for _ in range(500):
                env.step()
        if env.mj_viewer is not None:
            while env.mj_viewer.is_running():
                env.mj_viewer.sync()
                time.sleep(0.03)
    except KeyboardInterrupt:
        print("仿真已结束。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
