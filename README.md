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
Dockerfile.rebuild         基于已验收镜像覆盖 /app 的增量构建文件
tools/                     SwanLab 记录、PyTRIO 评测、SFT 数据构造与训练脚本
experiments/               试运行记录与微调日志（finetune-log.md）
local_check.py             本地回环自检（A/B/C、桩模型或 PyTRIO 真模型）
```

## 方案要点

1. 每轮从父检查点续训，不重跑全量初始化。
2. 决策来自平台模型：每轮把历史指标、漂移状态交给 `rnd.model()`，得到
   `epochs` / `learning_rate` 与假设说明；JSON 解析失败时回退到保守默认值。
3. 数据集策略：A 预算 10 / epochs 下限 2；B 前 3 次采纳固定 5 轮 / 0.0005，
   之后停训保留父产物；C 预算 1 / 下限 2。
4. `safe_train`（A/C）先跑主方案，未超过父模型时自动减半 epochs 和学习率再跑一次，
   两次都不优于父模型则保留父产物，避免回退。
5. 训练异常时保留父产物并正常结束轮次；平台测量值才是唯一分数来源，
   选择理由写入 `finish`。

## 构建镜像

```sh
# 从已验收的 aiplus-harness:0.1 派生，只覆盖 /app
docker build -f Dockerfile.rebuild -t aiplus-harness:r7 .
docker run --rm aiplus-harness:r7      # 输出 describe JSON
```

平台通过 `/image-archives` 接收 `docker save` 导出的单个 Linux/amd64 未压缩 tar：

```sh
docker save -o harness-r7.tar aiplus-harness:r7
```

## 本地校验

仓库内不包含赛题数据包。默认读取 `Desktop/AI+/数据包A`、`数据包B_extracted`、
`数据包C_extracted`；也可把数据包路径作为第三个参数传入。

```sh
python local_check.py A 1          # 子命令：数据集、轮数、可选数据包路径
python local_check.py B 4
python local_check.py C 2 "<绝对路径>/数据包C_extracted"
LOCAL_MODEL=pytrio python local_check.py A 1   # 用真实 Qwen 驱动
```

关键检查项：`describe` 输出一行 JSON、每轮至少一次真实模型调用、候选目录只含普通文件。

## 实验记录

```sh
pip install swanlab
swanlab login
python tools/log_swanlab.py --dataset B --records <state_out/records.json> --model <基础模型名>
```

## 提交信息

- 模型类型：平台白名单基础模型 Qwen/Qwen3.8-27B（二次确认）
- 实验记录：https://swanlab.cn/@1839229675/aiplus-harness
- 镜像：`docker save` 归档，在平台 "Harness 镜像" 页上传并选为就绪
- 源码：本仓库
- 文档：`docs/作品文档.md`
