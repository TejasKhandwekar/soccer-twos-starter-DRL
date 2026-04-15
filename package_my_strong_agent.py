import argparse
import glob
import os
import shutil


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


def main():
    parser = argparse.ArgumentParser(description="Copy latest checkpoint into my_strong_agent package.")
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
    args = parser.parse_args()

    experiment_dir = os.path.expanduser(args.experiment_dir)
    ckpt_file = find_latest_checkpoint(experiment_dir)
    ckpt_dir = os.path.dirname(ckpt_file)
    ckpt_metadata = f"{ckpt_file}.tune_metadata"

    params_pkl = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.exists(params_pkl):
        params_pkl = os.path.join(ckpt_dir, "..", "params.pkl")
    if not os.path.exists(params_pkl):
        raise FileNotFoundError("Could not find params.pkl near selected checkpoint")

    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_strong_agent")
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

    if args.metadata_out:
        metadata_out = os.path.expanduser(args.metadata_out)
        metadata_dir = os.path.dirname(metadata_out)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(metadata_out, "w") as f:
            f.write(f"selected_checkpoint={ckpt_file}\n")
            f.write(f"packaged_checkpoint={dst_ckpt_file}\n")
        print(f"Metadata written to: {metadata_out}")


if __name__ == "__main__":
    main()
