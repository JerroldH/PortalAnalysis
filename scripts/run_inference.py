"""
CLI: Run Booth inference for one or more patients.

Examples::

    python scripts/run_inference.py --mode csv --patient-id P001 --processed-dir N:/Booth_Processed --with-symptoms
    python scripts/run_inference.py --mode csv --patient-id P001 --distances-path path/to/right_finger_tapping_distances.csv
    python scripts/run_inference.py --mode csv --patient-id P001 --distances-path path/to/right_finger_tapping_distances.csv --source-video-path path/to/right_finger_tapping.mp4
    python scripts/run_inference.py --mode pose --patient-id P001 --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode pose --patient-id P001 --processed-dir N:/Booth_Processed --pose-path path/to/right_finger_tapping.csv
    python scripts/run_inference.py --mode video --patient-id P001 --raw-dir N:/.../Booth --processed-dir N:/Booth_Processed
    python scripts/run_inference.py --mode video --patient-id P001 --processed-dir N:/Booth_Processed --video-path path/to/right_finger_tapping.mp4
    python scripts/run_inference.py --mode video --patient-id P001 --processed-dir N:/Booth_Processed --video-path path/to/recording.mp4 --task finger_tapping --hand left
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
        help="csv: distances CSVs; pose: MediaPipe pose CSVs; video: raw MP4s "
        "(each mode supports explicit file paths like --video-path)",
    )
    p.add_argument("--patient-id", nargs="+", required=True, metavar="ID")
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["finger_tapping", "hand_open_close", "hand_up_down", "both_still"],
        default=None,
        help="Subset of tasks (csv/pose/raw-dir, or filter when inferring from filename)",
    )
    p.add_argument(
        "--task",
        choices=["finger_tapping", "hand_open_close", "hand_up_down", "both_still"],
        default=None,
        help="Explicit task for --video-path / --pose-path / --distances-path "
        "when the filename is not a known stem",
    )
    p.add_argument(
        "--hand",
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
        help="MP4 path for video mode (repeatable). Task/side from filename, or --task + --hand.",
    )
    p.add_argument(
        "--distances-path",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="Distances CSV for csv mode (repeatable). Task/side from filename, or --task + --hand.",
    )
    p.add_argument(
        "--source-video-path",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="Source MP4 for csv mode clinical timing metadata (repeatable; pair with --distances-path).",
    )
    p.add_argument(
        "--pose-path",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="Pose CSV for pose mode (repeatable). Task/side from filename, or --task + --hand.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for per-recording JSON scores (default: "
        "<processed-dir>/results/inference/)",
    )
    p.add_argument(
        "--model-version",
        default="latest",
        help="Model version subdirectory under models/<task>/ (default: latest)",
    )
    p.add_argument(
        "--with-symptoms",
        action="store_true",
        help="Also predict four clinical motor signs per task "
        "(amplitude reduction, sequence effect, slowness, halt/hesitation)",
    )
    p.add_argument("--video-width", type=int, default=1920)
    p.add_argument("--video-height", type=int, default=1080)
    return p.parse_args()


def main():
    args = parse_args()
    batch = BatchInferencePipeline(
        model_version=args.model_version,
        with_symptoms=args.with_symptoms,
    )

    if args.mode == "csv":
        if args.distances_path:
            if len(args.patient_id) != 1:
                print(
                    "ERROR: use exactly one --patient-id when using --distances-path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                entries = BatchInferencePipeline.entries_from_distances_paths(
                    patient_id=args.patient_id[0],
                    distances_paths=args.distances_path,
                    source_video_paths=args.source_video_path,
                    tasks=args.tasks,
                    hands=args.hand,
                    task=args.task,
                )
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            if not entries:
                print(
                    "ERROR: no distances CSVs matched --tasks / --hand filter.",
                    file=sys.stderr,
                )
                sys.exit(1)
            df = batch.run_from_distances_paths(
                entries=entries,
                processed_dir=args.processed_dir,
            )
        else:
            df = batch.run_from_csvs(
                patient_ids=args.patient_id,
                distances_dir=args.processed_dir,
                tasks=args.tasks,
                hands=args.hand,
            )
    elif args.mode == "pose":
        if args.pose_path:
            if len(args.patient_id) != 1:
                print(
                    "ERROR: use exactly one --patient-id when using --pose-path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                entries = BatchInferencePipeline.entries_from_pose_paths(
                    patient_id=args.patient_id[0],
                    pose_paths=args.pose_path,
                    tasks=args.tasks,
                    hands=args.hand,
                    task=args.task,
                )
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            if not entries:
                print(
                    "ERROR: no pose CSVs matched --tasks / --hand filter.",
                    file=sys.stderr,
                )
                sys.exit(1)
            df = batch.run_from_pose_paths(
                entries=entries,
                processed_dir=args.processed_dir,
                video_width=args.video_width,
                video_height=args.video_height,
            )
        else:
            df = batch.run_from_poses(
                patient_ids=args.patient_id,
                processed_dir=args.processed_dir,
                tasks=args.tasks,
                hands=args.hand,
                video_width=args.video_width,
                video_height=args.video_height,
            )
    else:
        if args.video_path:
            if len(args.patient_id) != 1:
                print(
                    "ERROR: use exactly one --patient-id when using --video-path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                entries = BatchInferencePipeline.entries_from_video_paths(
                    patient_id=args.patient_id[0],
                    video_paths=args.video_path,
                    tasks=args.tasks,
                    hands=args.hand,
                    task=args.task,
                )
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            if not entries:
                print(
                    "ERROR: no videos matched --tasks / --hand filter (or no --video-path given).",
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
                patient_ids=args.patient_id,
                raw_video_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                tasks=args.tasks,
                hands=args.hand,
                video_width=args.video_width,
                video_height=args.video_height,
            )

    print("\n" + df.to_string(index=False))
    BatchInferencePipeline.save_results(
        df,
        args.processed_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
