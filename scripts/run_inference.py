"""
CLI: Run Booth inference for one or more patients.

Examples::

    python scripts/run_inference.py --mode csv --patient-ids P001 --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode video --patient-ids P001 --raw-dir N:/.../Booth --processed-dir N:/Booth_Processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from portal_analysis.inference.batch import BatchInferencePipeline


def parse_args():
    p = argparse.ArgumentParser(description="PortalAnalysis — hand movement inference")
    p.add_argument("--mode", choices=["csv", "video"], default="csv")
    p.add_argument("--patient-ids", nargs="+", required=True, metavar="ID")
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["finger_tapping", "hand_open_close", "hand_up_down"],
        default=None,
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--model-version",
        default="latest",
        help="Model version subdirectory under models/<task>/ (default: latest)",
    )
    p.add_argument("--video-width", type=int, default=1920)
    p.add_argument("--video-height", type=int, default=1080)
    return p.parse_args()


def main():
    args = parse_args()
    batch = BatchInferencePipeline(model_version=args.model_version)

    if args.mode == "csv":
        df = batch.run_from_csvs(
            patient_ids=args.patient_ids,
            distances_dir=args.processed_dir,
            tasks=args.tasks,
        )
    else:
        if args.raw_dir is None:
            print("ERROR: --raw-dir is required for video mode.", file=sys.stderr)
            sys.exit(1)
        df = batch.run_from_videos(
            patient_ids=args.patient_ids,
            raw_video_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            tasks=args.tasks,
            video_width=args.video_width,
            video_height=args.video_height,
        )

    print("\n" + df.to_string(index=False))
    output = args.output or args.processed_dir / "results" / "inference.csv"
    BatchInferencePipeline.save_results(df, output)


if __name__ == "__main__":
    main()
