"""Meta-train and evaluate the FiLM personalization prototype."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personalization.experiment import (  # noqa: E402
    RANDOM_SEEDS,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes-per-epoch", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        nargs="+",
        default=list(RANDOM_SEEDS),
        help="At least five seeds for repeated calibration sampling.",
    )
    parser.add_argument(
        "--timevae-epochs",
        type=int,
        default=0,
        help="Optional conditional TimeVAE training epochs; zero disables it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[Step 1] Loading and splitting the frozen canonical CNN-LSTM.")
    print("[Step 2] Using seeded parametric users; TimeVAE is experimental future work.")
    print("[Step 3] Meta-training only the UserEncoder on train-subject episodes.")
    print("[Step 4-5] Evaluating held-out users with repeated calibration seeds.")
    reproduction_command = subprocess.list2cmdline(
        [
            sys.executable,
            "scripts\\run_film_personalization.py",
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--episodes-per-epoch",
            str(args.episodes_per_epoch),
            "--validation-episodes",
            str(args.validation_episodes),
            "--patience",
            str(args.patience),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--calibration-seeds",
            *[str(seed) for seed in args.calibration_seeds],
            "--timevae-epochs",
            str(args.timevae_epochs),
        ]
    )
    output_dir = run_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        episodes_per_epoch=args.episodes_per_epoch,
        epochs=args.epochs,
        validation_episodes=args.validation_episodes,
        patience=args.patience,
        seeds=tuple(args.calibration_seeds),
        finetune_epochs=args.finetune_epochs,
        timevae_epochs=args.timevae_epochs,
        command=reproduction_command,
    )
    print("[Step 6] Exported the frozen and preconditioned FiLM TFLite heads.")
    print(f"Results saved under: {output_dir}")


if __name__ == "__main__":
    main()
