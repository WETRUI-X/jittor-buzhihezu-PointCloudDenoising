#!/usr/bin/env python
"""
Select the best checkpoint using only the local validation set.

This script evaluates checkpoints on the project's validate_dataset and ranks
them by validation loss. Lower val_loss is better. It does not need official GT
data from the competition organizer.

Example:
    python select_best_checkpoint.py --ckpt_dir experiments/vm --copy_best

Useful quick test:
    python select_best_checkpoint.py \
        --ckpt_dir experiments/vm \
        --limit 3
"""

import argparse
import contextlib
import csv
import json
import random
import re
import shutil
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


MODEL_TARGET_FALLBACK = {
    "vm": "VelocityModule",
}


@dataclass
class CheckpointResult:
    checkpoint: str
    epoch: Optional[int]
    score: Optional[float]
    status: str
    error: str = ""


def checkpoint_epoch(path: Path) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.pkl$)", path.name)
    return int(match.group(1)) if match else None


def natural_key(path: Path) -> Tuple[int, str]:
    epoch = checkpoint_epoch(path)
    return (epoch if epoch is not None else -1, path.name)


def iter_checkpoints(
    ckpt_dir: Path,
    pattern: str,
    start_epoch: Optional[int],
    end_epoch: Optional[int],
) -> List[Path]:
    checkpoints = sorted(ckpt_dir.glob(pattern), key=natural_key)
    selected = []
    for ckpt in checkpoints:
        epoch = checkpoint_epoch(ckpt)
        if start_epoch is not None and (epoch is None or epoch < start_epoch):
            continue
        if end_epoch is not None and (epoch is None or epoch > end_epoch):
            continue
        selected.append(ckpt)
    return selected


def load_yaml(path: Path) -> Dict:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("omegaconf is required. Install it with: pip install omegaconf") from exc

    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return cfg


def config_path(config_dir: str, name: str) -> Path:
    path = Path(config_dir) / name
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    return path


def ensure_model_target(model_config: Dict, component_name: str) -> Dict:
    cfg = deepcopy(model_config)
    if "__target__" not in cfg:
        fallback = MODEL_TARGET_FALLBACK.get(component_name)
        if fallback is None:
            raise SystemExit(
                f"configs/model/{component_name}.yaml has no __target__, "
                "and no fallback target is known for it."
            )
        cfg["__target__"] = fallback
    return cfg


def ensure_system_target(system_config: Dict, component_name: str) -> Dict:
    cfg = deepcopy(system_config)
    cfg.setdefault("__target__", component_name)
    return cfg


def item_value(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def write_results(results: Iterable[CheckpointResult], output_dir: Path) -> None:
    rows = [asdict(item) for item in results]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "checkpoint_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "epoch", "score", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "checkpoint_ranking.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_existing_results(output_dir: Path) -> Dict[str, CheckpointResult]:
    json_path = output_dir / "checkpoint_ranking.json"
    if not json_path.exists():
        return {}

    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    results = {}
    for row in rows:
        item = CheckpointResult(**row)
        results[item.checkpoint] = item
    return results


def build_validation_context(args) -> Dict:
    import jittor as jt
    import numpy as np

    from src.data.dataset import DatasetConfig
    from src.data.transform import Transform
    from src.model.parse import get_model

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = load_yaml(Path(args.task_template))
    components = task.get("components")
    if not isinstance(components, dict):
        raise SystemExit(f"{args.task_template} must contain a components mapping.")

    for name in ["data", "transform", "model", "system"]:
        if name not in components:
            raise SystemExit(f"{args.task_template} is missing components.{name}.")

    data_path = Path(args.data_config) if args.data_config else config_path("configs/data", components["data"])
    transform_path = config_path("configs/transform", components["transform"])
    model_path = config_path("configs/model", components["model"])
    system_path = config_path("configs/system", components["system"])

    data_config = load_yaml(data_path)
    validate_config = data_config.get("validate_dataset")
    if validate_config is None:
        raise SystemExit(f"No validate_dataset found in {data_path}")

    transform_config = load_yaml(transform_path)
    model_config = ensure_model_target(load_yaml(model_path), components["model"])
    system_config = ensure_system_target(load_yaml(system_path), components["system"])

    validate_dataset_config = DatasetConfig.parse(**validate_config).split_by_cls()

    # get_model mutates model_config by deleting __target__, so always pass a copy.
    temp_model = get_model(
        model_config=deepcopy(model_config),
        transform_config=deepcopy(transform_config),
    )
    validate_transform = temp_model.get_validate_transform()
    if validate_transform is None:
        validate_transform = Transform.parse(**transform_config.get("validate_transform", {}))

    return {
        "seed": args.seed,
        "task": task,
        "model_config": model_config,
        "transform_config": transform_config,
        "system_config": system_config,
        "validate_dataset_config": validate_dataset_config,
        "validate_transform": validate_transform,
    }


def evaluate_checkpoint(ckpt_path: Path, context: Dict, log_path: Path) -> Tuple[Optional[float], Dict[str, float]]:
    import jittor as jt
    import numpy as np

    from src.data.dataset import PCDatasetModule
    from src.model.parse import get_model
    from src.system.parse import get_system

    jt.set_global_seed(context["seed"])
    np.random.seed(context["seed"])
    random.seed(context["seed"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            # get_model and get_system mutate config dicts, so pass fresh copies.
            model = get_model(
                model_config=deepcopy(context["model_config"]),
                transform_config=deepcopy(context["transform_config"]),
            )
            model.load(str(ckpt_path))
            model.set_predict(False)
            model.eval()

            dataset_module = PCDatasetModule(
                process_fn=model._process_fn,
                train_dataset_config=None,
                validate_dataset_config=context["validate_dataset_config"],
                predict_dataset_config=None,
                train_transform=None,
                validate_transform=context["validate_transform"],
                predict_transform=None,
                debug=False,
            )

            system = get_system(
                dataset_module=dataset_module,
                model=model,
                optimizer_config=None,
                loss_config=context["task"].get("loss"),
                trainer_config=None,
                writer=None,
                **deepcopy(context["system_config"]),
            )

            validate_dataloader = dataset_module.validate_dataloader()
            if validate_dataloader is None:
                raise RuntimeError("validate_dataloader is None")

            losses: List[float] = []
            system.on_validation_epoch_start()
            loaders = validate_dataloader if isinstance(validate_dataloader, dict) else {"validate": validate_dataloader}

            with jt.no_grad():
                for loader_name, dataloader in loaders.items():
                    for batch in dataloader:
                        system.on_validation_batch_start()
                        loss = system.validation_step(batch)
                        loss_float = item_value(loss)
                        losses.append(loss_float)
                        print(f"{loader_name}: loss={loss_float:.8f}")
                        system.on_validation_batch_end()

            system.on_validation_epoch_end()
            val_loss = mean(losses)
            metrics = {
                name: value
                for name, values in system._validation_loss.items()
                for value in [mean([float(v) for v in values])]
                if value is not None
            }
            if val_loss is not None:
                metrics["val/loss_mean"] = val_loss

            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            if hasattr(jt, "gc"):
                jt.gc()

    return metrics.get("val/loss_mean"), metrics


def _chamfer_distance(a, b) -> float:
    """与 evaluate.py 一致的 CD：双向最近点平方距离均值（点云已归一化）。"""
    from scipy.spatial import cKDTree

    tree_b = cKDTree(b)
    dist_a2b, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dist_b2a, _ = tree_a.query(b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def _normalize_unit_sphere(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    pc = pc - (p_max + p_min) / 2
    scale = np.sqrt((pc ** 2).sum(axis=1).max()).max()
    return pc / scale


def evaluate_checkpoint_cd(ckpt_path: Path, context: Dict, log_path: Path, args) -> Tuple[Optional[float], Dict[str, float]]:
    """在验证集上直接计算 Chamfer Distance：动态加噪 → 模型推理 → 对比 clean GT。

    与 loss 模式不同，该指标与竞赛评分直接对齐。噪声按固定种子生成，
    所有 checkpoint 面对完全相同的含噪输入，排名可比。
    """
    import jittor as jt
    import numpy as np

    from src.model.parse import get_model

    np.random.seed(context["seed"])
    random.seed(context["seed"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model = get_model(
                model_config=deepcopy(context["model_config"]),
                transform_config=deepcopy(context["transform_config"]),
            )
            model.load(str(ckpt_path))
            model.set_predict(True)
            model.eval()

            cds: List[float] = []
            sample_index = 0
            for cls, ds_config in context["validate_dataset_config"].items():
                for lazy_asset in ds_config.datapath.get_data():
                    if args.cd_limit is not None and sample_index >= args.cd_limit:
                        break
                    asset = lazy_asset.load()
                    pool = asset.sampled_vertices
                    if pool is None or pool.shape[0] < args.cd_points:
                        continue

                    rs = np.random.RandomState(context["seed"] + sample_index)
                    clean = pool[rs.choice(pool.shape[0], args.cd_points, replace=False)]
                    clean = _normalize_unit_sphere(clean.astype(np.float32))
                    noise_std = rs.uniform(args.noise_std_min, args.noise_std_max)
                    noise = rs.laplace(0.0, noise_std / np.sqrt(2.0), size=clean.shape)
                    noisy = (clean + noise).astype(np.float32)

                    pred = model.predict_step({"pc_noisy": jt.array(noisy[None])})
                    denoised = pred[0]["pc_denoised"]
                    cd = _chamfer_distance(denoised, clean)
                    cd_noisy = _chamfer_distance(noisy, clean)
                    cds.append(cd)
                    print(f"sample {sample_index} ({asset.path}): cd={cd:.8f} cd_noisy={cd_noisy:.8f} improve={100*(1-cd/cd_noisy):.2f}")
                    sample_index += 1

            mean_cd = mean(cds)
            metrics = {"val/cd_mean": mean_cd} if mean_cd is not None else {}
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            if hasattr(jt, "gc"):
                jt.gc()

    return metrics.get("val/cd_mean"), metrics


def rank_results(results: List[CheckpointResult]) -> List[CheckpointResult]:
    ok = [item for item in results if item.status == "ok" and item.score is not None]
    bad = [item for item in results if item.status != "ok" or item.score is None]
    return sorted(ok, key=lambda item: item.score) + bad


def run_selection(args, checkpoints: List[Path], existing: Dict[str, CheckpointResult]) -> List[CheckpointResult]:
    output_dir = Path(args.output_dir)
    context = build_validation_context(args)
    results: List[CheckpointResult] = []

    for index, ckpt in enumerate(checkpoints, start=1):
        ckpt_abs = ckpt.resolve()
        ckpt_key = str(ckpt)

        if ckpt_key in existing and existing[ckpt_key].status == "ok":
            print(f"[{index}/{len(checkpoints)}] skip existing: {ckpt}")
            results.append(existing[ckpt_key])
            continue

        epoch = checkpoint_epoch(ckpt)
        log_path = output_dir / "logs" / f"{ckpt.stem}_{args.metric}.log"

        print(f"[{index}/{len(checkpoints)}] validate {args.metric}: {ckpt}")
        try:
            if args.metric == "cd":
                score, _ = evaluate_checkpoint_cd(ckpt_abs, context, log_path, args)
            else:
                score, _ = evaluate_checkpoint(ckpt_abs, context, log_path)
        except Exception as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("\n\nException traceback:\n")
                traceback.print_exc(file=log_file)
            error = f"validation failed, see {log_path}: {exc}"
            print(f"  {error}")
            results.append(CheckpointResult(ckpt_key, epoch, None, "validate_failed", error))
            write_results(rank_results(results), output_dir)
            continue

        if score is None:
            error = f"could not compute validation {args.metric}, see {log_path}"
            print(f"  {error}")
            results.append(CheckpointResult(ckpt_key, epoch, None, "parse_failed", error))
        else:
            print(f"  {args.metric}: {score:.8f}")
            results.append(CheckpointResult(ckpt_key, epoch, score, "ok"))

        write_results(rank_results(results), output_dir)

    return rank_results(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best checkpoint by local validation loss.")
    parser.add_argument("--metric", choices=["loss", "cd"], default="loss",
                        help="loss=训练损失代理; cd=验证集上真实 Chamfer Distance，与竞赛评分对齐")
    parser.add_argument("--cd_points", type=int, default=32768, help="CD 模式每个模型采样点数")
    parser.add_argument("--cd_limit", type=int, default=None, help="CD 模式最多评估多少个验证模型")
    parser.add_argument("--noise_std_min", type=float, default=0.005, help="CD 模式加噪 std 下限")
    parser.add_argument("--noise_std_max", type=float, default=0.020, help="CD 模式加噪 std 上限")
    parser.add_argument("--ckpt_dir", default="experiments/vm", help="Directory containing checkpoint_*.pkl files.")
    parser.add_argument("--pattern", default="checkpoint_*.pkl", help="Checkpoint filename pattern.")
    parser.add_argument("--task_template", default="configs/task/train_vm.yaml", help="Training task yaml.")
    parser.add_argument("--data_config", default="", help="Optional data yaml. Defaults to the task component data config.")
    parser.add_argument("--output_dir", default="checkpoint_selection", help="Directory for logs and rankings.")
    parser.add_argument("--use_cuda", type=int, default=1, help="Jittor CUDA flag.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--start_epoch", type=int, default=None, help="Only evaluate checkpoints with epoch >= this value.")
    parser.add_argument("--end_epoch", type=int, default=None, help="Only evaluate checkpoints with epoch <= this value.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many checkpoints after filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip checkpoints already marked ok in checkpoint_ranking.json.")
    parser.add_argument("--copy_best", action="store_true", help="Copy the best checkpoint to output_dir/best_checkpoint.pkl.")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {ckpt_dir}")

    checkpoints = iter_checkpoints(ckpt_dir, args.pattern, args.start_epoch, args.end_epoch)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {ckpt_dir / args.pattern}")

    existing = load_existing_results(output_dir) if args.resume else {}
    ranked = run_selection(args, checkpoints, existing)
    write_results(ranked, output_dir)

    ok_ranked = [item for item in ranked if item.status == "ok" and item.score is not None]
    if not ok_ranked:
        print("No checkpoint was evaluated successfully.")
        return 1

    best = ok_ranked[0]
    print("\nBest checkpoint")
    print(f"  checkpoint: {best.checkpoint}")
    print(f"  epoch: {best.epoch}")
    print(f"  {args.metric}: {best.score:.8f}")
    print(f"  ranking: {output_dir / 'checkpoint_ranking.csv'}")

    if args.copy_best:
        best_path = output_dir / "best_checkpoint.pkl"
        shutil.copy2(best.checkpoint, best_path)
        print(f"  copied best checkpoint to: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
