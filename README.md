# GRPO-Baseline

本仓库旨在基于 [veRL](https://github.com/volcengine/verl) 框架和，从零搭建一套支持大语言模型 (LLM) 进行 GRPO (Group Relative Policy Optimization) 训练的基础实验范式，主要面向 Math 和 Coding 推理任务。

当前阶段的目标不是一次性做很复杂的算法扩展，而是先跑通，搭建合适的后续实验平台

# 初始化项目

## 依赖安装

- 提示：目前demo阶段主要是使用AMD官方打包好的docker镜像，后续可能需要自己打包，此步也可跳过
- 手动安装（目前会用到的包）
- 自动安装（占位）

## 设置环境变量

- 复制`.env.example`并重命名为`.env`，填入各种变量（eg：HF_TOKEN)
- 说明
  - 目前主要是方便下载模型
  - 没有HF_TOKEN的话记得去HuggingFace创建一个，记得保存在自己本地
  - ！！！重要凭证不要随代码一并提交

## 下载模型/数据集

```bash
# [常用命令示例]
# 1. 搜索模型/数据集:
python download.py search Qwen --limit 20
python download.py search gsm8k --type dataset --limit 5

# 2. 下载模型 (默认下载到 {DEFAULT_MODEL_DIR}，没有会创建):
python download.py download Qwen/Qwen2.5-1.5B-Instruct

# 3. 下载数据集 (默认下载到 {DEFAULT_DATA_DIR}):
python download.py download openai/gsm8k --type dataset

# 4. 指定目录下载:
python download.py download Qwen/Qwen2.5-1.5B-Instruct --local-dir tmp
```

## 设置 Wandb

- Wandb 是一个用于监控训练进程、管理训练的很好的开源实现
- 在 Wandb 上注册一个账号，联系 @Qiuyc 添加进 Team

# veRL 的使用

## AMD GPU

- 这一步主要是参考AMD官方提供的教程，可跳到 @ Demo 部分直接看

### 使用已预装 verl 的预构建 Docker 镜像

1. 拉取 Docker 镜像

```bash
docker pull rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev
```

1. 启动并连接到 Docker 容器

```bash
docker run --rm -it --device /dev/dri --device /dev/kfd -p 8265:8265 --group-add video \
--cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME/.ssh:/root/.ssh \
-v $HOME:$HOME --shm-size 128G -w $PWD --name rocm_verl \
rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev /bin/bash
```

### TODO：后续开发可能需要修改veRL源码

## TODO：Nvidia GPU

# 运行实验环境

- 在AMD服务器上的运行命令已经打包好了，直接运行以下的命令：

```bash
chmod +x run_docker.sh
./run_docker.sh
```

- 它主要做了这么几件事：
  - 自动删除同名旧容器
  - 挂载 `/dev/dri` 和 `/dev/kfd`，用于 ROCm/AMD GPU 访问
  - 添加 `video` 组权限
  - 开启 `SYS_PTRACE`、`seccomp=unconfined`、`--privileged`
  - 使用 `--ipc=host` 和 `--shm-size 128G`
  - 映射端口 8265:8265（Ray的默认端口）
  - 挂载：
    - `$HOME/.ssh:/root/.ssh:ro`
    - `$HOME:$HOME`
    - 工作目录
    - checkpoint 目录（包含数据集）
  - 透传环境变量：
    - `WANDB_API_KEY`
    - `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`
- 这个脚本设置了自动推出，如果在容器内 `exit` 退出，容器会自动关闭

# 启动 Demo
- 进入容器后，运行 `./train` 目录下的脚本，例如：
```bash
./train/example.sh
```
- 这会运行一个 ppo 训练脚本，使用 Qwen2-1.5B，在 gsm8k 上训练，14分钟就能跑完