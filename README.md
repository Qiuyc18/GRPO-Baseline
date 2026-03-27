# GRPO-Baseline

本仓库旨在基于 [veRL](https://github.com/volcengine/verl) 框架和，从零搭建一套支持大语言模型 (LLM) 进行 GRPO (Group Relative Policy Optimization) 训练的基础实验范式，主要面向 Math 和 Coding 推理任务。

# 初始化项目
```bash
chmod +x ./setup.sh && ./setup.sh
```

# veRL 的安装
## AMD GPU
### 使用已预装 verl 的预构建 Docker 镜像
1. 拉取 Docker 镜像
```bash
docker pull rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev
```
2. 启动并连接到 Docker 容器
```bash
docker run --rm -it --device /dev/dri --device /dev/kfd -p 8265:8265 --group-add video \
--cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME/.ssh:/root/.ssh \
-v $HOME:$HOME --shm-size 128G -w $PWD --name rocm_verl \
rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev /bin/bash
```
### TODO：后续开发可能需要修改veRL源码

## TODO：Nvidia GPU
