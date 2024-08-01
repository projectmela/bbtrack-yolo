"""
A basic example to evaluate trackers using "TrackEval" package, (already integrated with
`bbtrack_yolo` package).

Pitfalls:
1. While tracking, if "trackeval/trackers/.../tracker_name/seq_name.txt" already exists,
    it will skip tracking that sequence.

Steps:
1. Load detection files from "predictions" folder
2. Track detections with multiple trackers
"""

from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from boxmot import BYTETracker, BoTSORT
from tqdm.auto import tqdm

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBoxTracker import TrackerType, BBoxTracker
from bbtrack_yolo.custom_bb2023_mot import CustomBB2023MOT
from bbtrack_yolo.util import get_default_tqdm_args


def get_device():
    """Automatically select device for torch based on environment / hardware"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        # Default fallback to CPU
        return "cpu"


def track_dets_with_tracker(
    tracker: TrackerType,
    dets_paths: List[Path],
    trackers_split_dir: Union[Path, str],
    tqdm_args: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    verbose: bool = True,
    overwrite: bool = False,
):
    """Track multiple sequence detections with tracker and save to TrackEval folder
    :param tracker: tracker instance to track, should be initialized
    :param dets_paths: List of detection paths
    :param trackers_split_dir: Path to save tracking results,
        e.g. "trackeval/trackers/BB2023/BB2023-test"
    :param tqdm_args: tqdm arguments
    :param progress_bar: Show progress bar
    :param verbose: Show verbose message
    :param overwrite: Overwrite existing tracking results
    """

    trackers_split_dir = Path(trackers_split_dir)
    tracker_name = f"{tracker}"

    # Prepare tqdm progress bar arguments
    tqdm_args = get_default_tqdm_args(tqdm_args)
    tqdm_args.update({"desc": f"Tracking with {tracker_name}"})

    # Track detections
    for dets_path in tqdm(
        iterable=dets_paths,
        disable=not progress_bar,
        **tqdm_args,
    ):

        # Remove prefix "YYYYMMDD-HHMMSS_" to get seq_name
        seq_name = dets_path.parent.name[len("YYYYMMDD-HHMMSS_") :]
        # Load detections
        dets = BBoxDetection.load_from(dets_path)
        # Remove boxes that have width or height less than 1
        dets = dets.filter_invalid_boxes()
        # dets = dets.filter_low_threshold(0.1)

        # Setup save path
        result_file_path = trackers_split_dir / f"{tracker_name}/data/{seq_name}.txt"
        result_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and result_file_path.exists():
            if verbose:
                tqdm.write(
                    f"skip '{seq_name}' with '{tracker_name}' since"
                    f" existing result is found ('{result_file_path}')."
                )
            continue

        if verbose:
            tqdm.write(f"{tracker_name} tracking {seq_name}")

        video_path = None
        if tracker.reid_model_name or "BoT" in tracker_name:
            video_name = dets_path.parent.name[len("YYYYMMDD-HHMMSS_") :]
            video_path = (
                dets_path.parent.parent.parent.parent / f"videos/{video_name}.MP4"
            )
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
        # Track detections
        trks = tracker.track(dets, progress_bar=progress_bar, video_path=video_path)

        # Save to prediction folder
        # trks.save_to(
        #     dets_path / f"track_{split}/{tracking_param}.parquet", overwrite=True
        # )

        # Save to TrackEval folder
        trks.save_to_mot17(result_file_path)


if __name__ == "__main__":
    freeze_support()

    split = "full"  # TrackEval package requires a split name to be specified
    trackers_split_dir = Path(f"trackeval/trackers/BB2023/BB2023-{split}")
    # all tracker result must be saved in this folder as:
    # trackers_split_dir / tracker_name / {seq_name}.txt

    # Get all detection paths
    dets_paths = list(
        Path(
            "predictions/"
            "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003"
        ).glob("*/detection.csv")
    )
    print(f"Found {len(dets_paths)} detection files")
    print("\n".join(map(str, dets_paths)))

    # Add all trackers
    trackers = [
        BBoxTracker(BYTETracker()),
        # BBoxTracker(
        #     BoTSORT(
        #         model_weights=None,
        #         device=get_device(),
        #         fp16=False,
        #         with_reid=False,
        #     ),
        # ),
        # BBoxTracker(
        #     BoTSORT(
        #         model_weights="resnet50_msmt17.pt",
        #         device=get_device(),
        #         fp16=False,
        #         with_reid=False,
        #     ),
        # ),
    ]

    # Print tracker names
    tracker_names = [str(tracker) for tracker in trackers]
    print("Tracking with following trackers:")
    print("\n".join(tracker_names))

    # Track detections with multiple trackers
    for tracker in trackers:
        track_dets_with_tracker(
            tracker=tracker,
            dets_paths=dets_paths,
            trackers_split_dir=trackers_split_dir,
            progress_bar=True,
            verbose=True,
            overwrite=False,
        )

    # Evaluate tracker results in "trackeval" folder
    dataset_config = CustomBB2023MOT.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": "trackeval/gt/BB2023",
            "TRACKERS_FOLDER": "trackeval/trackers/BB2023",
            "TRACKERS_TO_EVAL": None,  # None to eval all trackers
            "SPLIT_TO_EVAL": split,  # Valid: 'train', 'test', 'full'
        }
    )
    eval_config = CustomBB2023MOT.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": True,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": True,
        }
    )

    data_eval = CustomBB2023MOT(dataset_config, eval_config)

    res = data_eval.evaluate()
