import functools
import itertools
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from loguru import logger
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBTracker import (
    BBTracker,
    BYTETrackerConfig,
    BBBotSortTracker,
    BotSortTrackerConfig,
)
from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.custom_bb2023_mot import CustomBB2023MOT
from bbtrack_yolo.util import get_default_tqdm_args


# tracking param options
def generate_tracking_param_list(
    tracker_name: str, param_options: Dict[str, List]
) -> Iterable[BYTETrackerConfig]:
    """Generate tracking parameters for tuning"""
    for values in itertools.product(*param_options.values()):  # type: ignore
        params = dict(zip(param_options.keys(), values))
        if tracker_name == "BYTETracker":
            yield BYTETrackerConfig(**params)
        elif tracker_name == "BotSortTracker":
            yield BotSortTrackerConfig(**params)
        else:
            raise ValueError(
                f"Unsupported tracker name: {tracker_name}"
                f"Supported names: 'BYTETracker', 'BotSortTracker'"
            )


def track_dets_with_tracking_param(
    tracking_param: Union[BYTETrackerConfig, BotSortTrackerConfig],
    dets_paths: List[Path],
    trackers_split_dir: Union[Path, str],
    tqdm_args: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    verbose: bool = True,
    overwrite: bool = False,
):
    """Track detections with given tracking config and save to TrackEval folder
    :param tracking_param: Tracking parameter
    :param dets_paths: List of detection paths
    :param trackers_split_dir: Path to save tracking results,
        e.g. "trackeval/trackers/BB2023/BB2023-test"
    :param tqdm_args: tqdm arguments
    :param progress_bar: Show progress bar
    :param verbose: Show verbose message
    :param overwrite: Overwrite existing tracking results
    """

    trackers_split_dir = Path(trackers_split_dir)

    # prepare progress bar
    tqdm_args = get_default_tqdm_args(tqdm_args)
    tqdm_args.update({"desc": f"Tracking with {tracking_param}"})
    dets_path_loop = dets_paths
    if progress_bar:
        # TODO: report mypy error to tqdm
        dets_path_loop = tqdm(  # type: ignore
            iterable=dets_paths,
            **tqdm_args,
        )

    # track detections
    for dets_path in dets_path_loop:

        # remove prefix "YYYYMMDD-HHMMSS_" to get seq_name
        seq_name = dets_path.parent.name[len("YYYYMMDD-HHMMSS_") :]
        tracker_name = f"{tracking_param}"

        dets = BBoxDetection.load_from(dets_path)

        if isinstance(tracking_param, BotSortTrackerConfig):
            tracker = BBBotSortTracker(config=tracking_param)
        elif isinstance(tracking_param, BYTETrackerConfig):
            tracker = BBTracker(config=tracking_param)
        else:
            raise ValueError(f"Unsupported tracking parameter: {tracking_param}")

        tracker_file_dest = trackers_split_dir / f"{tracker_name}/data/{seq_name}.txt"
        tracker_file_dest.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and tracker_file_dest.exists():
            if verbose:
                tqdm.write(
                    f"skip '{seq_name}' with '{tracker_name}' since"
                    f" it has been tested (found '{tracker_file_dest}')."
                )
            continue

        if verbose:
            tqdm.write(f"Tracking {seq_name} with {tracker_name}")

        trks = tracker.track(dets, progress_bar=True)

        # save to prediction folder
        # trks.save_to(
        #     dets_path / f"track_{split}/{tracking_param}.parquet", overwrite=True
        # )

        # save to TrackEval folder
        trks.save_to_mot17(tracker_file_dest)

    return tracker_names


def batch_track_in_parallel(
    dets_paths: List[Path],
    tracking_param_list: List[BYTETrackerConfig],
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
        track_dets_with_tracking_param,
        dets_paths=dets_paths,
        trackers_split_dir=trackers_split_dir,
        progress_bar=False,
        verbose=False,
    )

    process_map(single_proc_func, tracking_param_list, max_workers=n_proc, chunksize=1)


def evaluate_trackers_with_trackeval(
    split: str,
    trackers: Optional[List[str]] = None,
):
    """Evaluate tracking with TrackEval"""
    # evaluate tracking
    dataset_config = CustomBB2023MOT.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": "trackeval/gt/BB2023",
            "TRACKERS_FOLDER": "trackeval/trackers/BB2023",
            "TRACKERS_TO_EVAL": trackers,  # None to eval all trackers
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


if __name__ == "__main__":
    freeze_support()

    split = "full"
    split_ratio = 0.7
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

    # generate tracking parameter options for trackers
    # # default tracking parameter
    param_options: Dict[str, List] = {
        "track_high_thresh": [0.5],
        "track_low_thresh": [0.1],
        "new_track_thresh": [0.6],
        "track_buffer": [30],
        "match_thresh": [0.8],
    }

    # test options
    # param_options: Dict[str, List] = {
    #     "track_high_thresh": [0.3, 0.5],
    #     "track_low_thresh": [0.05, 0.1],
    #     "new_track_thresh": [0.3, 0.8],
    #     "track_buffer": [30, 50],
    #     "match_thresh": [0.7, 0.9],
    # }

    # all options
    # param_options: Dict[str, List] = {
    #     "track_high_thresh": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    #     "track_low_thresh": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    #     "new_track_thresh": [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8],
    #     "track_buffer": [5, 10, 30, 50, 100],
    #     "match_thresh": [0.7, 0.75, 0.8, 0.85, 0.9],
    # }
    tracker_name = "BotSortTracker"
    # tracker_name = "BYTETracker"

    tracking_param_list = list(
        generate_tracking_param_list(
            tracker_name=tracker_name,
            param_options=param_options,
        )
    )
    tracker_names = [str(tracking_param) for tracking_param in tracking_param_list]

    track_dets_with_tracking_param(
        tracking_param=tracking_param_list[0],
        dets_paths=dets_paths,
        trackers_split_dir=trackers_split_dir,
        progress_bar=True,
        verbose=True,
        overwrite=False,
    )

    # track detections with tracking parameters in parallel
    # batch_track_in_parallel(
    #     dets_paths=dets_paths,
    #     tracking_param_list=tracking_param_list,
    #     trackers_split_dir=trackers_split_dir,
    #     n_proc=90,
    # )

    logger.info(f"Evaluating trackers: {tracker_names}")
    evaluate_trackers_with_trackeval(
        split=split,
        trackers=tracker_names,
    )
