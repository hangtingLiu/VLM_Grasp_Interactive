# VLM_Grasp_Interactive

> 基于参考文章进行环境配置与项目搭建。

Ubuntu 22.04 / RTX 5060 用户请参阅 [复现说明](REPRODUCE.zh-CN.md)，其中包含 CUDA 12.8、自定义算子编译、DeepSeek 视觉接口和 MuJoCo 抓放验证。

[![Project Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/graspnet/graspnet-baseline)  
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

---

> **参考文章：**  
> https://blog.csdn.net/agentssl/article/details/148089323

> **相关视频：**  
> [【mujoco】多模态大模型协助UR机械臂实现抓取任务](https://www.bilibili.com/video/BV1r4pFz5EZ9/?share_source=copy_web&vd_source=66cd731dbf7f013478ae50f3c9609159)

---

## 📖 目录

- [1. 配置环境](#1-配置环境)
- [2. 创建主文件夹](#2-创建主文件夹)
- [3. 移植 manipulator_grasp 操作包](#3-移植-manipulator_grasp-操作包)
- [4. 下载 graspnet-baseline 项目](#4-下载-graspnet-baseline-项目)
- [5. 安装 Torch 及其他环境包](#5-安装-torch-及其他环境包)
- [6. 编译并安装 PointNet++ 自定义算子](#6-编译并安装-pointnet-自定义算子)
- [7. 编译并安装基于 PyTorch CUDA 的 k-NN 算子](#7-编译并安装基于-pytorch-cuda-的-k-nn-算子)
- [8. 安装 graspnetAPI 工具包](#8-安装-graspnetapi-工具包)
- [9. 下载 GraspNet-baseline 权重](#9-下载-graspnet-baseline-权重)
- [10. 继续完善环境](#10-继续完善环境)

---

## 1. 配置环境

```bash
conda create -n vlm_graspnet python=3.10
conda activate vlm_graspnet
```

---

## 2. 创建主文件夹

```text
VLM_Grasp_Interactive
```

---

## 3. 移植 `manipulator_grasp` 操作包

主文件夹中移植下面项目的 `manipulator_grasp` 操作包（进去找这个文件夹）：

```bash
git clone https://gitee.com/chaomingsanhua/manipulator_grasp.git
```

---

## 4. 下载 `graspnet-baseline` 项目

主文件夹中下载 `graspnet-baseline` 项目：

```bash
git clone https://github.com/graspnet/graspnet-baseline.git
```

打开 `graspnet-baseline`，修改其中的 `requirements.txt`：

```bash
cd graspnet-baseline
sudo gedit requirements.txt
```

保存后执行安装指令：

```bash
pip install -r requirements.txt
```

---

## 5. 安装 Torch 及其他环境包

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.1
```

---

## 6. 编译并安装 `PointNet++` 自定义算子

> **说明：**  
> `PointNet++` 是处理 3D 点云的经典神经网络，需要自定义 CUDA 算子实现高效采样和特征聚合。  
> `VoteNet` 是基于 `PointNet++` 的 3D 目标检测框架。

```bash
cd pointnet2
python setup.py install
cd ../
```

---

## 7. 编译并安装基于 PyTorch CUDA 的 `k-NN` 算子

- `k-NN` 算子：用于快速计算数据点之间最近邻关系的算法模块
- CUDA 加速：利用 NVIDIA 显卡进行并行计算加速
- PyTorch 扩展：为深度学习框架添加自定义 C++/CUDA 操作

```bash
cd knn
python setup.py install
cd ../
```

---

## 8. 安装 `graspnetAPI` 工具包

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
```

修改 `setup.py`，将 `sklearn` 替换为 `scikit-learn`：

```bash
sudo gedit setup.py
```

修改后开始安装：

```bash
pip install .
cd ../
```

---

## 9. 下载 `GraspNet-baseline` 权重

下载 `GraspNet-baseline` 的权重（训练好的）。

在 `VLM_Grasp_Interactive` 文件夹下新建 `logs/log_rs` 文件夹，并将下载好的权重 `checkpoint-rs.tar` 放进 `log_rs`。

---

## 10. 继续完善环境

> **提示：**  
> 可以运行代码后按需补装依赖，缺啥装啥；如果遇到版本兼容问题，再做对应修改。  
> 【每个人有自己的玄学体质，后续考虑 docker 打包】

```bash
pip install opencv-python
pip install --force-reinstall "numpy==1.26.4
pip install ultralytics==8.3.98
pip install "opencv-python==4.5.5.64" --force-reinstall
pip install openai-whisper
pip install soundfile
pip install sounddevice
pip install pydub
pip install openai
# 进入 graspnetAPI 目录并安装 pip install -e .
pip install --upgrade transforms3d
```


