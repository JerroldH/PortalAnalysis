"""
CLI: Run Booth inference for one or more patients.

Examples::

    python scripts/run_inference.py --mode csv --patient-ids P001 --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode pose --patient-ids P001 --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode video --patient-ids P001 --raw-dir N:/.../Booth --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode video --patient-ids P001 --processed-dir N:/Booth_Processed --video-path path/to/right_finger_tapping.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from portal_analysis.inference.batch import BatchInferencePipeline


def parse_args():
    p = argparse.ArgumentParser(description="PortalAnalysis — hand movement inference")
    p.add_argument(
        "--mode",
        choices=["csv", "pose", "video"],
        default="csv",
        help="csv: distances CSVs; pose: MediaPipe pose CSVs; video: raw MP4s",
    )
    p.add_argument("--patient-ids", nargs="+", required=True, metavar="ID")
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["finger_tapping", "hand_open_close", "hand_up_down"],
        default=None,
    )
    p.add_argument(
        "--hands",
        choices=["left", "right", "both"],
        default="both",
        help="Which hand(s) to run: left, right, or both (default: both)",
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument(
        "--video-path",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="MP4 path for video mode (repeatable). Task/subtask inferred from filename.",
    )
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
            hands=args.hands,
        )
    elif args.mode == "pose":
        df = batch.run_from_poses(
            patient_ids=args.patient_ids,
            processed_dir=args.processed_dir,
            tasks=args.tasks,
            hands=args.hands,
            video_width=args.video_width,
            video_height=args.video_height,
        )
    else:
        if args.video_path:
            if len(args.patient_ids) != 1:
                print(
                    "ERROR: use exactly one --patient-ids when using --video-path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                entries = BatchInferencePipeline.entries_from_video_paths(
                    patient_id=args.patient_ids[0],
                    video_paths=args.video_path,
                    tasks=args.tasks,
                    hands=args.hands,
                )
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            if not entries:
                print(
                    "ERROR: no videos matched --tasks / --hands filter (or no --video-path given).",
                    file=sys.stderr,
                )
                sys.exit(1)
            df = batch.run_from_video_paths(
                entries=entries,
                processed_dir=args.processed_dir,
                video_width=args.video_width,
                video_height=args.video_height,
            )
        elif args.raw_dir is None:
            print(
                "ERROR: video mode requires --raw-dir or at least one --video-path.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            df = batch.run_from_videos(
                patient_ids=args.patient_ids,
                raw_video_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                tasks=args.tasks,
                hands=args.hands,
                video_width=args.video_width,
                video_height=args.video_height,
            )

    print("\n" + df.to_string(index=False))
    output = args.output or args.processed_dir / "results" / "inference.csv"
    BatchInferencePipeline.save_results(df, output)


if __name__ == "__main__":
    main()
