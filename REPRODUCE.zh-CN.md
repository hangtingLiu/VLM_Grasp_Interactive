# Ubuntu 22.04 与 RTX 5060 复现说明

在 Ubuntu 22.04.5、RTX 5060 Laptop GPU（8 GB）、NVIDIA 驱动 615.71.09 和 GCC/G++ 11.4 上验证。流程为 MuJoCo 相机取图 → DeepSeek 根据指令定位目标 → SAM 分割 → GraspNet 生成候选 → MuJoCo 验证并执行抓取与放置。

## 1. 创建环境

在 Ubuntu 文件系统中克隆仓库并进入项目根目录。当前仓库已包含机械臂场景、GraspNet 源码和 `logs/log_rs/checkpoint-rs.tar`。

```bash
conda create -n vlm_graspnet --override-channels -c conda-forge python=3.10 pip -y
conda activate vlm_graspnet
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -n vlm_graspnet --override-channels \
  -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit=12.8.1 -y
```

`nvidia-smi` 验证驱动，`nvcc --version` 应显示 CUDA 12.8。以下设置用于本次编译：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=2
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.ones(1, device='cuda'))"
```

## 2. 安装依赖和编译扩展

```bash
python -m pip install ninja setuptools wheel "Cython<3" \
  "numpy==1.26.4" "pandas==2.2.3" "mujoco==3.3.1" \
  "spatialmath-python==1.1.14" "spatialgeometry==1.1.0" \
  "roboticstoolbox-python==1.1.1" "modern-robotics==1.1.1" \
  "opencv-python==4.11.0.86" "ultralytics==8.3.98" openai

(cd graspnet-baseline/pointnet2 && python -m pip install . --no-build-isolation)
(cd graspnet-baseline/knn && python -m pip install . --no-build-isolation)
python -m pip install ./graspnet-baseline/graspnetAPI \
  "numpy==1.26.4" "opencv-python==4.11.0.86" --no-build-isolation
python -c "import torch; import pointnet2._ext; from knn_pytorch import knn_pytorch; from graspnetAPI import GraspGroup; print('扩展与 API 可用')"
```

固定 `spatialgeometry==1.1.0` 是为了匹配 Robotics Toolbox 1.1.1 的 `SceneGroup` 接口。编译 k-NN 时应让 PyTorch 能看到 GPU，否则该扩展可能只编译 CPU 后端。GraspNet API 的 ROS 1/catkin 提示只影响其 ROS 转换接口，仿真可继续运行。

## 3. 配置视觉模型

将 API 密钥保存到仓库之外的单行文本文件。复制示例配置，再将其中的 `api_key_file` 改为密钥文件路径：

```bash
cp .vlm.example.json .vlm.local.json
```

也可在没有本地配置时使用 `VLM_BASE_URL`、`VLM_MODEL` 和 `VLM_API_KEY` 环境变量。如果 DeepSeek 可以直连且代理配置导致客户端报错，可在本地 JSON 中设置 `"trust_env": false`。首次分割会下载 `sam_b.pt`。视觉服务会收到当前仿真画面和文字指令。

## 4. 运行并验证

在桌面终端运行：

```bash
python main_vlm.py "抓苹果" --verify-body Apple
python main_vlm.py "抓锤子" --verify-body Hammer
```

`--verify-body` 是可选的测试断言。它检查视觉结果是否对应预期仿真刚体，省略后可直接输入其他物体的指令。无窗口测试可使用：

```bash
MUJOCO_GL=egl python main_vlm.py "我需要一把刀" --headless --verify-body Knife
```

程序先用 GraspNet 和当前深度图生成候选，再在后台模拟并过滤桌面和邻物接触。夹空会提前退出，后台最多试 12 个候选或 20 秒，以先到者为准。这个时间不包含视觉 API 请求和模型推理。通过后桌面窗口会播放相应动作，终端会打印抬升高度和落点偏差。

验证过苹果、香蕉和锤子的文字抓放，以及刀的语音目标定位和保存掩码下的抓放。将仿真中的刀平移 10 厘米并旋转 25 度后，也用仿真掩码单独验证了抓放。候选由当前点云计算，能否找到有效姿态取决于分割、摆放和夹爪可达性。

## 5. 可选中文语音模式

Ubuntu 需要 `parec`、`paplay`（由 `pulseaudio-utils` 提供）和带普通话音色的 `espeak-ng`。在独立 conda 环境安装 Whisper：

```bash
python -m pip install openai-whisper
python main_vlm.py --voice
```

首次识别会将 Whisper base 模型缓存到 `outputs/whisper`。在录音提示后说出指令，核对识别结果后按回车，或直接键入更正文本。使用本机默认麦克风和音频输出。验证过实际录音驱动苹果抓放。不同麦克风和环境下的识别效果可能不同。
