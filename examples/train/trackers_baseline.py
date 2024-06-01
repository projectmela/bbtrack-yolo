"""This script uses all trackers with default parameters to track detections 
and evaluate with TrackEval
"""

import functools
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from boxmot import BYTETracker, OCSORT, StrongSORT, HybridSORT, BoTSORT
from loguru import logger
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBoxTracker import TrackerType, BBoxTracker
from bbtrack_yolo.util import get_default_tqdm_args
from examples.train.batch_tracking import evaluate_trackers_with_trackeval


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

        # Track detections
        trks = tracker.track(dets, progress_bar=progress_bar)

        # Save to prediction folder
        # trks.save_to(
        #     dets_path / f"track_{split}/{tracking_param}.parquet", overwrite=True
        # )

        # Save to TrackEval folder
        trks.save_to_mot17(result_file_path)


def batch_parallel_track_dets_with_trackers(
    dets_paths: List[Path],
    trackrs_list: List[TrackerType],
    trackers_split_dir: Union[Path, str],
    n_proc: int = 1,
):
    """Batch track with multiprocessing
    :param dets_paths: List of detection paths
    :param tracking_param_list: List of tracking parameters
    :param trackers_split_dir: Path to save tracking results
    :param n_proc: Number of processes to use
    """

    trackers_split_dir = Path(trackers_split_dir)

    single_proc_func = functools.partial(
        track_dets_with_tracker,
        dets_paths=dets_paths,
        trackers_split_dir=trackers_split_dir,
        progress_bar=False,
        verbose=True,
    )

    process_map(
        single_proc_func,
        trackrs_list,
        desc="Batch tracking in parallel",
        bar_format="{l_bar}{bar:10}{r_bar}",
        max_workers=n_proc,
        chunksize=1,
    )


if __name__ == "__main__":
    freeze_support()

    split = "full"
    trackers_split_dir = Path(f"trackeval/trackers/BB2023/BB2023-{split}")

    # get all detection paths
    dets_paths = list(
        Path(
            "predictions/"
            "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003"
        ).glob("*/detection.csv")
    )
    print(f"Found {len(dets_paths)} detection files")
    print("\n".join(map(str, dets_paths)))

    # TODO: DEBUG: only keep 0176.seg_1
    # dets_paths = [p for p in dets_paths if "0176.seg_1" in p.parent.name]

    reid_model_names = [
        # "clip_vehicleid.pt",  # latest clip model
        # "resnet50_msmt17.pt",  # classic ReID model
        "osnet_x1_0_msmt17.pt",  # Same-domain ReID best perf
        "osnet_ain_x1_0_msmt17.pt",  # Multi-source domain generalization best perf
        "lmbn_n_duke.pt",
        "osnet_x0_25_msmt17.pt",
    ]

    get_tracker_list = [
        lambda: BBoxTracker(BYTETracker()),
        lambda: BBoxTracker(OCSORT()),
        lambda: BBoxTracker(
            BoTSORT(
                model_weights=None,
                device="cpu",
                fp16=False,
                with_reid=False,
            )
        ),
    ]

    for reid_model_name in reid_model_names:
        get_tracker_list.extend(
            [
                lambda: BBoxTracker(
                    BoTSORT(
                        model_weights=Path(reid_model_name),
                        device="cuda:0",
                        fp16=False,
                        with_reid=True,
                    ),
                    reid_model_name=reid_model_name,
                ),
                lambda: BBoxTracker(
                    HybridSORT(
                        reid_weights=Path(reid_model_name),
                        device="cuda:0",
                        half=False,
                        det_thresh=0.5,
                    ),
                    reid_model_name=reid_model_name,
                ),
                # lambda: BBoxTracker(
                #     StrongSORT(
                #         model_weights=Path(reid_model_name),
                #         device="cuda:0",
                #         fp16=False,
                #     ),
                #     reid_model_name=reid_model_name,
                # ),
            ]
        )

    for get_tracker in get_tracker_list:
        track_dets_with_tracker(
            tracker=get_tracker(),
            dets_paths=dets_paths,
            trackers_split_dir=trackers_split_dir,
            progress_bar=True,
            verbose=True,
            overwrite=False,
        )

    tracker_names = [str(get_tracker()) for get_tracker in get_tracker_list]
    logger.info(f"Evaluating trackers: {tracker_names}")
    evaluate_trackers_with_trackeval(
        split=split,
        trackers=tracker_names,
    )
