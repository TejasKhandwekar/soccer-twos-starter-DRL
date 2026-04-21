import argparse
import glob
import os
import shutil
import zipfile


def find_latest_checkpoint(experiment_dir: str) -> str:
    pattern = os.path.join(experiment_dir, "**", "checkpoint_*", "checkpoint-*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under: {experiment_dir}")

    def step_num(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split("-")[-1])
        except Exception:
            return -1

    candidates.sort(key=step_num)
    return candidates[-1]


def find_checkpoint_by_step(experiment_dir: str, step: int) -> str:
    pattern = os.path.join(
        experiment_dir,
        "**",
        f"checkpoint_{step:06d}",
        f"checkpoint-{step}",
    )
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for step {step} found under: {experiment_dir}"
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Copy a checkpoint from a Ray experiment into an agent package folder."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to a Ray experiment dir, e.g. ~/scratch/ray_results/PPO_STRONG_SELFPLAY_SHAPED",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default=None,
        help="Optional output file path for selected checkpoint metadata",
    )
    parser.add_argument(
        "--package-dir",
        type=str,
        default="my_strong_agent",
        help="Target package directory (default: my_strong_agent)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Optional checkpoint step to package (e.g., 1600). Defaults to latest.",
    )
    parser.add_argument(
        "--zip-output",
        type=str,
        default=None,
        help="Optional zip path to create after packaging",
    )
    args = parser.parse_args()

    experiment_dir = os.path.expanduser(args.experiment_dir)
    if args.checkpoint_step is None:
        ckpt_file = find_latest_checkpoint(experiment_dir)
    else:
        ckpt_file = find_checkpoint_by_step(experiment_dir, args.checkpoint_step)

    ckpt_dir = os.path.dirname(ckpt_file)
    ckpt_metadata = f"{ckpt_file}.tune_metadata"

    params_pkl = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.exists(params_pkl):
        params_pkl = os.path.join(ckpt_dir, "..", "params.pkl")
    if not os.path.exists(params_pkl):
        raise FileNotFoundError("Could not find params.pkl near selected checkpoint")

    pkg_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.expanduser(args.package_dir)
    )
    dst_ckpt_dir = os.path.join(pkg_dir, "checkpoint")
    os.makedirs(dst_ckpt_dir, exist_ok=True)

    dst_ckpt_file = os.path.join(dst_ckpt_dir, "checkpoint")
    shutil.copy2(ckpt_file, dst_ckpt_file)
    if os.path.exists(ckpt_metadata):
        shutil.copy2(ckpt_metadata, f"{dst_ckpt_file}.tune_metadata")
    else:
        print(
            "Warning: checkpoint .tune_metadata not found. "
            "Restore may fail for some RLlib versions."
        )
    shutil.copy2(params_pkl, os.path.join(dst_ckpt_dir, "params.pkl"))

    print(f"Selected checkpoint: {ckpt_file}")
    print(f"Packaged checkpoint: {dst_ckpt_file}")

    if args.zip_output:
        zip_output = os.path.expanduser(args.zip_output)
        zip_dir = os.path.dirname(zip_output)
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)

        with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(pkg_dir):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, os.path.dirname(pkg_dir))
                    zf.write(full_path, rel_path)
        print(f"Created zip: {zip_output}")

    if args.metadata_out:
        metadata_out = os.path.expanduser(args.metadata_out)
        metadata_dir = os.path.dirname(metadata_out)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(metadata_out, "w") as f:
            f.write(f"selected_checkpoint={ckpt_file}\n")
            f.write(f"packaged_checkpoint={dst_ckpt_file}\n")
            f.write(f"package_dir={pkg_dir}\n")
        print(f"Metadata written to: {metadata_out}")


if __name__ == "__main__":
    main()
