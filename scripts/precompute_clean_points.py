"""Pre-sample clean point clouds from the official training meshes."""

import argparse
import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm


def sample_seed(global_seed: int, relative_path: Path) -> int:
    key = f"{global_seed}:{relative_path.as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())
        if not geometries:
            raise ValueError("mesh scene contains no geometry")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh has no vertices or faces")
    return mesh


def atomic_save(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as file:
            temporary = file.name
            np.save(file, points, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def process_one(task: Tuple[str, str, str, int, int, bool]) -> Tuple[str, str]:
    input_root_s, output_root_s, relative_s, num_points, seed, overwrite = task
    input_root = Path(input_root_s)
    output_root = Path(output_root_s)
    relative = Path(relative_s)
    source = input_root / relative / "models" / "model_normalized.obj"
    destination = output_root / relative / "clean.npy"

    if destination.exists() and not overwrite:
        return "skipped", relative.as_posix()

    try:
        mesh = load_mesh(source)
        # trimesh uses NumPy's global RNG internally. A stable per-file seed
        # makes output independent of task scheduling and worker count.
        np.random.seed(sample_seed(seed, relative))
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (num_points, 3):
            raise ValueError(f"unexpected sampled shape: {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("sampled points contain non-finite values")
        atomic_save(destination, points)
        return "written", relative.as_posix()
    except Exception as exc:
        return "failed", f"{relative.as_posix()}: {type(exc).__name__}: {exc}"


def discover_meshes(input_root: Path):
    for path in sorted(input_root.glob("shapenet/*/*/models/model_normalized.obj")):
        yield path.parent.parent.relative_to(input_root)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="dataset_train", help="Official mesh dataset root.")
    parser.add_argument("--output_dir", default="dataset_train_pcd", help="Clean point cache root.")
    parser.add_argument("--num_points", type=int, default=50000, help="Surface points saved per mesh.")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16), help="Worker processes.")
    parser.add_argument("--seed", type=int, default=123, help="Global deterministic seed.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing clean.npy files.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many meshes (for testing).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if args.num_points <= 0:
        raise SystemExit("--num_points must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not input_root.is_dir():
        raise SystemExit(f"input directory does not exist: {input_root}")

    discovered = list(discover_meshes(input_root))
    if args.limit is not None:
        discovered = discovered[:args.limit]
    if not discovered:
        raise SystemExit(f"no model_normalized.obj files found under: {input_root}")

    tasks = [
        (str(input_root), str(output_root), str(relative), args.num_points, args.seed, args.overwrite)
        for relative in discovered
    ]
    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures = []

    if args.workers == 1:
        results = map(process_one, tasks)
        for status, message in tqdm(results, total=len(tasks), desc="Sampling meshes"):
            counts[status] += 1
            if status == "failed":
                failures.append(message)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(process_one, tasks, chunksize=1)
            for status, message in tqdm(results, total=len(tasks), desc="Sampling meshes"):
                counts[status] += 1
                if status == "failed":
                    failures.append(message)

    print(f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']}")
    for failure in failures:
        print(f"ERROR: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
