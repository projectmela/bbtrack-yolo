import functools
from pathlib import Path
from typing import List

from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig
from bbtrack_yolo.BBoxDetection import BBoxDetection


def track_dets_with_config(config: BYTETrackerConfig, det_file: Path):
    """Track detections from a file path with given parameters"""
    dets = BBoxDetection.load_from(det_file)
    tracker = BBTracker(config=config)
    trks = tracker.track(dets, progress_bar=False)
    return trks


def batch_track_dets_with_config(
    config: BYTETrackerConfig,
    det_files: List[Path],
    n_proc: int = 1,
):
    """Batch track detections from a list of file paths with given parameters"""
    single_proc_func = functools.partial(track_dets_with_config, config)
    process_map(single_proc_func, det_files, max_workers=n_proc)


if __name__ == "__main__":

    prediction_dir = Path("predictions")

    # detection files are usually "predictions/<model_name>/<sequence_name>/detection.csv"
    dets_files = list(prediction_dir.glob("*/*/detection.csv"))

    print("Found detection files:")
    print("\n".join([str(df) for df in dets_files]))

    tracker_config = BYTETrackerConfig(
        # threshold for the first association
        track_high_thresh=0.5,
        # threshold for the second association
        track_low_thresh=0.03,
        # threshold for init new track if the detection does not match any tracks
        new_track_thresh=0.1,
        # buffer to calculate the time when to remove tracks
        track_buffer=30,
        # threshold for matching tracks
        match_thresh=0.8,
    )

    batch_track_dets_with_config(tracker_config, dets_files, n_proc=4)