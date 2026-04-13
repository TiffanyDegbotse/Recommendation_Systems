# AI-assisted code generation (Claude, Anthropic) – https://claude.ai
"""
setup.py
========
Orchestrates the full Visual Vibe pipeline from a blank working directory:

    Step 1 — make_dataset  : download data, parse metadata, build splits
    Step 2 — build_features: extract naive + classical audio features
    Step 3 — model         : train classifiers, build retrieval index, evaluate

Run this once to go from nothing to fully trained models:

    python setup.py               # full ~30 GB mel-spectrogram download
    python setup.py --quick       # smoke-test with ~2 GB of data
    python setup.py --from-step 2 # skip download, re-run features + training
    python setup.py --from-step 3 # skip download + features, re-train only

Note: This is a project setup script, NOT a Python packaging setup.py.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(
    label: str,
    module: str,
    extra_args: list[str] | None = None,
) -> None:
    """
    Execute a pipeline step as a subprocess and exit on failure.

    Args:
        label:      Human-readable step name for progress output.
        module:     Python module path (e.g. 'scripts.make_dataset').
        extra_args: Additional CLI arguments to forward to the module.
    """
    extra_args = extra_args or []
    cmd        = [sys.executable, "-m", module] + extra_args

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start  = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌  {label} failed (exit code {result.returncode}). "
              f"Check the output above.")
        sys.exit(result.returncode)

    print(f"\n✅  {label} complete  ({elapsed:.1f}s)")


def check_prerequisites() -> None:
    """Warn if required directories or files are missing before starting."""
    issues = []
    if not Path("scripts").is_dir():
        issues.append("  - scripts/ directory not found. "
                      "Run from the project root.")
    for module in ["scripts/make_dataset.py",
                   "scripts/build_features.py",
                   "scripts/model.py"]:
        if not Path(module).exists():
            issues.append(f"  - {module} not found.")

    if issues:
        print("⚠️  Prerequisites missing:")
        for issue in issues:
            print(issue)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Visual Vibe pipeline end-to-end.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Download only 2 mel-spectrogram shards (~2 GB) for a smoke-test.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the mel-spectrogram download (data already on disk).")
    parser.add_argument(
        "--from-step", type=int, default=1, choices=[1, 2, 3],
        help="Resume pipeline from this step "
             "(1=download, 2=features, 3=train). Default: 1.")
    parser.add_argument(
        "--force-features", action="store_true",
        help="Re-extract features even if caches exist (passed to build_features).")
    parser.add_argument(
        "--processed-dir", default="./data/processed",
        help="Directory for processed data (default: ./data/processed).")
    parser.add_argument(
        "--models-dir", default="./models",
        help="Directory for trained models (default: ./models).")
    args = parser.parse_args()

    check_prerequisites()

    print("\n🎵  Visual Vibe — Pipeline Setup")
    print(f"    Quick mode      : {args.quick}")
    print(f"    Skip download   : {args.skip_download}")
    print(f"    Starting at step: {args.from_step}")

    # ── Step 1: Download data & build splits ──────────────────────────────────
    if args.from_step <= 1:
        step1_args = []
        if args.quick:
            step1_args.append("--quick")
        if args.skip_download:
            step1_args.append("--skip-download")

        run_step(
            label     = "Step 1 / 3 — Download data & build splits",
            module    = "scripts.make_dataset",
            extra_args=step1_args,
        )

    # ── Step 2: Extract audio features ───────────────────────────────────────
    if args.from_step <= 2:
        step2_args = [f"--processed-dir={args.processed_dir}"]
        if args.force_features:
            step2_args.append("--force")

        run_step(
            label     = "Step 2 / 3 — Extract audio features",
            module    = "scripts.build_features",
            extra_args=step2_args,
        )

    # ── Step 3: Train models & evaluate ──────────────────────────────────────
    if args.from_step <= 3:
        run_step(
            label     = "Step 3 / 3 — Train classifiers & build retrieval index",
            module    = "scripts.model",
            extra_args=[
                f"--processed-dir={args.processed_dir}",
                f"--models-dir={args.models_dir}",
            ],
        )

    print("\n" + "="*60)
    print("  🎉  Pipeline complete!")
    print(f"  Models saved to : {args.models_dir}/")
    print(f"  Run inference   : python main.py --track-id <ID>")
    print(f"                    python main.py --mood energetic")
    print("="*60 + "\n")
