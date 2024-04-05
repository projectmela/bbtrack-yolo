import functools
import itertools
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig
from bbtrack_yolo.custom_bb2023_mot import CustomBB2023MOT
from bbtrack_yolo.util import get_default_tqdm_args

split = "train"
# split = "test"
# split = "full"
split_ratio = 0.7
skip_tested = True
track_eval_dir = Path(f"data/trackeval/trackers/BB2023/BB2023-{split}")


# tracking param options
def tracking_params() -> Iterable[BYTETrackerConfig]:
    """Generate tracking parameters for tuning"""

    # all options
    param_options = {
        "track_high_thresh": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "track_low_thresh": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15],
        "new_track_thresh": [0.1, 0.2, 0.35, 0.5],
        "track_buffer": [1, 2, 5, 10, 30, 100],
        "match_thresh": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    }

    for values in itertools.product(*param_options.values()):  # type: ignore
        params = dict(zip(param_options.keys(), values))
        yield BYTETrackerConfig(**params)


def get_dets_list() -> List[BBoxDetection]:
    """Load detections"""
    search_dir = Path(
        "data/"
        "predictions/"
        "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003"
    )
    assert search_dir.exists()
    dets_list = [BBoxDetection.load_from(f) for f in search_dir.glob("*/detection.csv")]
    assert dets_list
    return dets_list


def track_dets_with_params(
    tracking_param: BYTETrackerConfig,
    dets_list: List[BBoxDetection],
    tqdm_args: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    verbose: bool = True,
):
    """Track detections with given parameters"""

    tqdm_args = get_default_tqdm_args(tqdm_args)
    loop = dets_list
    if progress_bar:
        # TODO: report mypy error to tqdm
        loop = tqdm(  # type: ignore
            iterable=dets_list,
            **tqdm_args,
        )

    for dets in loop:
        # save to track eval folder
        assert dets.folder is not None, "Detection folder is not set"
        seq_name = dets.folder.name[16:]

        tracker = BBTracker(config=tracking_param)
        track_eval_dest = track_eval_dir / f"{tracking_param}/data/{seq_name}.txt"
        if skip_tested and track_eval_dest.exists():
            if verbose:
                tqdm.write(
                    f"Skipping {dets.folder.name} with {tracking_param} since"
                    f" it has been tested (found '{track_eval_dest}')."
                )
            continue

        max_frame = dets.frame_range[1]
        split_frame = round(max_frame * split_ratio)

        if verbose:
            tqdm.write(f"Tracking {dets.folder.name} with {tracking_param}")
        # split detections by train/test
        if split == "train":
            trks = tracker.track(dets[:split_frame], progress_bar=False)
        elif split == "test":
            trks = tracker.track(dets[split_frame:], progress_bar=False)
        else:
            trks = tracker.track(dets, progress_bar=False)

        assert dets.folder is not None

        # save to prediction folder
        # trks.save_to(
        #     dets.folder / f"track_{split}/{tracking_param}.parquet", overwrite=True
        # )

        # save to TrackEval folder
        trks.save_to_mot17(track_eval_dest, overwrite=True)


def batch_track(
    tracking_params: List[BYTETrackerConfig],
    n_proc: int = 1,
):
    """Batch track with multiprocessing"""
    dets_list = get_dets_list()

    single_proc_func = functools.partial(
        track_dets_with_params, dets_list=dets_list, progress_bar=False, verbose=False
    )

    process_map(single_proc_func, tracking_params, max_workers=n_proc, chunksize=1)


def eval():
    # evaluate tracking
    dataset_config = CustomBB2023MOT.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": "data/trackeval/gt/BB2023",
            "TRACKERS_FOLDER": "data/trackeval/trackers/BB2023",
            "TRACKERS_TO_EVAL": None,  # None to eval all trackers
            "SPLIT_TO_EVAL": split,  # Valid: 'train', 'test', 'full'
        }
    )
    eval_config = CustomBB2023MOT.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": True,
        }
    )

    data_eval = CustomBB2023MOT(dataset_config, eval_config)

    res = data_eval.evaluate()


if __name__ == "__main__":
    freeze_support()

    batch_track(list(tracking_params()), n_proc=64)

    eval()
