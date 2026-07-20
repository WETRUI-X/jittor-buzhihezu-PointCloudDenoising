# 基于 Jittor 的点云降噪 Baseline

本项目用于点云降噪任务：输入含噪点云 `noisy.npy`，模型预测每个点的三维位移，并输出相同点数的 `denoised.npy`。项目保留官方 OBJ 训练流程，同时提供 clean point cloud 缓存训练流程，用于减少每个 epoch 重复解析 OBJ 和 mesh 表面采样造成的 CPU/IO 开销。

## 环境安装

推荐使用 Python 3.9，并确保 GCC/G++ 版本不高于 10。

```bash
conda create -n jittor2A python=3.9 -y
conda activate jittor2A
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
```

`requirements.txt` 包含：

- `jittor`
- `numpy`
- `trimesh`
- `scipy`
- `omegaconf`

如需运行 `evaluate.py` 的精确 P2S 计算，可额外安装：

```bash
pip install point-cloud-utils
```

## 数据准备

将官方训练集和测试集放在项目根目录：

```text
dataset_train/
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj

dataset_test_noisy/
└── shapenet/<synset_id>/<model_id>/noisy.npy
```

例如：

```bash
tar xzf dataset_train.tar.gz
unzip dataset_test_noisy.zip
```

`datalist/train.txt`、`datalist/validate.txt` 和 `datalist/test.txt` 中保存相对于数据集根目录的模型路径，例如：

```text
shapenet/04401088/d7ed512f7a7daf63772afc88105fa679
```

## 原始 OBJ 训练

原始 baseline 每次读取 OBJ，并动态执行 mesh 表面采样、归一化、加噪和 patch 构造：

```bash
python run.py --task configs/task/train_vm.yaml
```

权重默认保存在：

```text
experiments/vm/checkpoint_<epoch>.pkl
```

## Clean Point Cloud 缓存

### `precompute_clean_points.py` 的作用

`scripts/precompute_clean_points.py` 将官方训练集中的 OBJ mesh 预采样成 clean point cloud：

```text
dataset_train_pcd/
└── shapenet/<synset_id>/<model_id>/clean.npy
```

默认每个 mesh 采样 50000 个表面点，输出必须为：

```text
shape: (50000, 3)
dtype: np.float32
```

缓存只固定 clean 表面点。训练时仍会动态抽取 32768 点、重新添加随机噪声并重新构造 patch，因此不会固定 noisy 数据。

### 小规模测试

先测试一个模型：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd \
  --num_points 50000 \
  --workers 1 \
  --seed 123 \
  --limit 1
```

再测试前 100 个模型和多进程：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd \
  --num_points 50000 \
  --workers 8 \
  --seed 123 \
  --limit 100
```

### 生成完整缓存

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd \
  --num_points 50000 \
  --workers 16 \
  --seed 123
```

已有 `clean.npy` 默认会被跳过，因此中断后可直接重新运行同一命令。需要重新生成已有缓存时才使用 `--overwrite`：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd \
  --num_points 50000 \
  --workers 16 \
  --seed 123 \
  --overwrite
```

参数说明：

- `--input_dir`：官方 OBJ 训练集根目录。
- `--output_dir`：clean 点云缓存目录。
- `--num_points`：每个 mesh 保存的点数，默认 50000。
- `--workers`：CPU worker 数，建议先测试 8，再尝试 16。
- `--seed`：全局随机种子；每个模型使用稳定的独立种子。
- `--limit`：只处理前 N 个模型，适合功能测试。
- `--overwrite`：覆盖已存在的 `clean.npy`。

检查 OBJ 和缓存数量：

```bash
find dataset_train -path '*/models/model_normalized.obj' -type f | wc -l
find dataset_train_pcd -name clean.npy -type f | wc -l
```

完整数据中两者应当一致。当前官方训练集预期为 15833 个模型。

### 使用内存盘

如果 `/dev/shm` 空间充足，可以直接把缓存生成到内存盘：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir /dev/shm/dataset_train_pcd \
  --num_points 50000 \
  --workers 16 \
  --seed 123

ln -s /dev/shm/dataset_train_pcd dataset_train_pcd
```

创建软链接前，应确保项目根目录中不存在同名的 `dataset_train_pcd` 文件或目录。服务器关机或重启后，`/dev/shm` 内容通常会消失。

## 使用缓存训练

快速调试配置每个 epoch 使用 1000 个训练样本，batch size 保持 16：

```bash
python run.py --task configs/task/train_vm_cached_debug.yaml
```

正式缓存训练每个 epoch 使用 10000 个训练样本：

```bash
python run.py --task configs/task/train_vm_cached.yaml
```

缓存模式的数据流程为：

```text
clean.npy (50000 点)
  -> 动态随机抽取 32768 点
  -> 归一化
  -> 动态添加 Laplace 噪声
  -> 构造 1000 点局部 patch
  -> 训练 displacement/velocity target
```

原始 OBJ 配置没有被覆盖，仍可随时使用：

```bash
python run.py --task configs/task/train_vm.yaml
```

## 选择最佳 Checkpoint

### `select_best_checkpoint.py` 的作用

训练会生成多个 `checkpoint_<epoch>.pkl`。`select_best_checkpoint.py` 使用本地 `validate_dataset` 逐个计算 validation loss，并按 loss 从低到高排序，不需要比赛官方测试 GT。

需要注意：validation loss 是模型训练目标的代理指标，不等同于官方 CD/P2S 排名，但通常比直接固定使用最后一个 epoch 更可靠。

### 快速测试

仅评估前 3 个 checkpoint：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --limit 3
```

### 评估全部 checkpoint 并复制最佳权重

原始 OBJ 训练对应：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm.yaml \
  --output_dir checkpoint_selection \
  --copy_best
```

缓存训练对应：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --task_template configs/task/train_vm_cached.yaml \
  --output_dir checkpoint_selection_cached \
  --copy_best
```

输出包括：

```text
checkpoint_selection/
├── checkpoint_ranking.csv
├── checkpoint_ranking.json
├── best_checkpoint.pkl
└── logs/
```

常用参数：

- `--pattern`：checkpoint 文件匹配规则，默认 `checkpoint_*.pkl`。
- `--start_epoch` / `--end_epoch`：限制评估 epoch 范围。
- `--limit`：最多评估多少个 checkpoint。
- `--resume`：跳过排名 JSON 中已经成功评估的 checkpoint。
- `--copy_best`：复制最佳权重为 `best_checkpoint.pkl`。
- `--data_config`：显式指定用于验证的数据配置。
- `--use_cuda 0`：使用 CPU 验证；默认使用 CUDA。

例如只评估 epoch 80 至 99，并支持断点继续：

```bash
python select_best_checkpoint.py \
  --ckpt_dir experiments/vm \
  --start_epoch 80 \
  --end_epoch 99 \
  --resume \
  --copy_best
```

## 推理

修改 `configs/task/predict_vm.yaml` 中的权重路径：

```yaml
load_ckpt: checkpoint_selection/best_checkpoint.pkl
```

然后运行：

```bash
python run.py --task configs/task/predict_vm.yaml
```

预测配置使用独立的空 `predict_transform`，不会对已经含噪的 `noisy.npy` 再次添加噪声。结果保存到：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
```

每个输出应满足：

```text
shape 与输入 noisy.npy 完全相同
dtype 为 np.float32
```

## 验证预测输出

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

noisy_root = Path('dataset_test_noisy')
result_root = Path('results/dataset_test_noisy')
errors = []

for noisy_path in noisy_root.glob('shapenet/*/*/noisy.npy'):
    relative = noisy_path.relative_to(noisy_root)
    output_path = result_root / relative.parent / 'denoised.npy'
    if not output_path.exists():
        errors.append(f'缺少输出: {output_path}')
        continue

    noisy = np.load(noisy_path, mmap_mode='r')
    denoised = np.load(output_path, mmap_mode='r')
    if denoised.shape != noisy.shape:
        errors.append(f'shape 错误: {output_path}: {denoised.shape} != {noisy.shape}')
    if denoised.dtype != np.float32:
        errors.append(f'dtype 错误: {output_path}: {denoised.dtype}')
    if not np.isfinite(denoised).all():
        errors.append(f'包含 NaN/Inf: {output_path}')

if errors:
    print('\n'.join(errors))
    raise SystemExit(f'验证失败，共 {len(errors)} 个问题')
print('验证通过：所有 denoised.npy 的 shape、dtype 和数值均正常')
PY
```

## 打包提交

```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

最终压缩包结构必须是：

```text
result.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy
```
