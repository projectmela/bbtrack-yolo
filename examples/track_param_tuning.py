import itertools
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from bbtrack_yolo.BBTracker import BYTETrackerConfig, BBTracker
from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.custom_bb2023_mot import CustomBB2023MOT

split = "train"
# split = "test"
# split = "full"
split_ratio = 0.7
skip_tested = True


# tracking param options
def tracking_params() -> Iterable[BYTETrackerConfig]:
    """Generate tracking parameters for tuning"""

    param_options = {
        "track_high_thresh": [0.3, 0.5, 0.7],
        "track_low_thresh": [0.05, 0.1],
        "new_track_thresh": [0.05, 0.1],
        "track_buffer": [15, 30],
        "match_thresh": [0.6, 0.8],
    }

    for values in itertools.product(*param_options.values()):  # type: ignore
        params = dict(zip(param_options.keys(), values))
        yield BYTETrackerConfig(**params)


def track():
    # load detections
    track_eval_dir = Path(f"data/trackeval/trackers/BB2023/BB2023-{split}")
    search_dir = Path(
        "data/"
        "predictions/"
        "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003"
    )
    assert search_dir.exists()
    dets_list = [BBoxDetection.load_from(f) for f in search_dir.glob("*/detection.csv")]
    assert dets_list

    # tracking
    for params in tqdm(
        list(tracking_params()), desc="Param", bar_format="{l_bar}{bar:10}{r_bar}"
    ):
        tracker_name = f"{params}"
        for dets in tqdm(
            dets_list, desc="Tracking", bar_format="{l_bar}{bar:10}{r_bar}", leave=False
        ):
            # save to track eval folder
            seq_name = dets.folder.name[16:]

            tracker = BBTracker(config=params)
            track_eval_dest = track_eval_dir / f"{tracker_name}/data/{seq_name}.txt"
            if skip_tested and track_eval_dest.exists():
                tqdm.write(
                    f"Skipping {dets.folder.name} with {params} since"
                    f" it has been tested (found '{track_eval_dest}')."
                )
                continue

            max_frame = dets.frame_range[1]
            split_frame = round(max_frame * split_ratio)

            tqdm.write(f"Tracking {dets.folder.name} with {params}")
            if split == "train":
                trks = tracker.track(dets[:split_frame])
            elif split == "test":
                trks = tracker.track(dets[split_frame:])
            else:
                trks = tracker.track(dets)

            assert dets.folder is not None

            # save to prediction folder
            trks.save_to(
                dets.folder / f"track_{split}/{params}.parquet", overwrite=True
            )

            trks.save_to_mot17(track_eval_dest, overwrite=True)


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
    track()

    freeze_support()

    eval()
