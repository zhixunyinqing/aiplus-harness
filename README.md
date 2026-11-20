# aiplus-harness

"智训引擎"工业模型自进化赛题的 Harness 方案。镜像内提供 `describe` / `train` / `predict`
三个固定入口，在授权预算内对 CPU 工业模型做多轮续训、验证与产物选择。

## 目录

```text
app/
  harness.py               固定入口（describe / train / predict）
  contest_submission.py    平台回调：一轮训练 + 提交候选 + 预测
  contestlib/
    decision.py            组织上下文、解析模型返回的 JSON 决策
    runner.py              safe_train：主方案 + 回退方案，择优
    drift.py               训练/验证分布漂移统计
    state.py               跨轮 records 与检查点状态读写
  baselines/               父模型定义、数据读取、训练与预测实现
sdk/                       contest_harness SDK 1.0.4（构建镜像时安装）
Dockerfile                 平台镜像构建文件
tools/log_swanlab.py       把本地轮次指标写入 SwanLab 实验记录
```

## 方案要点

1. 每轮从父检查点续训，不重跑全量初始化。
2. 决策来自平台模型：每轮把历史指标、漂移状态交给 `rnd.model()`，得到
   `epochs` / `learning_rate` 与假设说明；JSON 解析失败时回退到保守默认值。
3. `safe_train` 先跑主方案，未超过父模型时自动减半 epochs 和学习率再跑一次，
   两次都不优于父模型则保留父产物，避免回退。
4. 平台测量值才是唯一分数来源，代码不另报分数；选择理由写入 `finish`。

## 构建镜像

```sh
docker build --platform linux/amd64 \
  --build-arg BASE_PYTHON=docker.m.daocloud.io/library/python:3.12-slim \
  --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t aiplus-harness:0.1 .
docker run --rm aiplus-harness:0.1
```

平台通过 `/image-archives` 接收 `docker save` 导出的单个 Linux/amd64 未压缩 tar：

```sh
docker save -o aiplus-harness-image.tar aiplus-harness:0.1
```

## 本地校验

仓库内不包含赛题数据包。把公开的 `train.jsonl` / `validation.jsonl` 放到本地目录后，
可复用 `local_frame.py` 的帧构造与回环宿主跑通 `train` 与 `predict` 协议。
关键检查项：`describe` 输出一行 JSON、每轮至少一次真实模型调用、候选目录只含普通文件。

## 实验记录

```sh
pip install swanlab
swanlab login
python tools/log_swanlab.py --dataset B --records <state_out/records.json> --model <基础模型名>
```

## 提交信息

- 模型类型：平台白名单基础模型（二次确认）
- 镜像：`docker save` 归档，在平台 "Harness 镜像" 页上传并选为就绪
- 源码：本仓库
- 文档：`docs/作品文档.md`
