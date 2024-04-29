import argparse
import functools
from pathlib import Path
from typing import List

from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBDetecor import BBDetector, BBDetectorConfig
from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig
from bbtrack_yolo.BBoxDetection import BBoxDetection


def track_dets_with_config(config: BYTETrackerConfig, dets_file: Path):
    """Track detections from a file path with given parameters"""
    dets = BBoxDetection.load_from(dets_file)
    tracker = BBTracker(config=config)
    trks = tracker.track(dets, progress_bar=False)
    trks.save_to(dets_file.parent / "tracks.csv")
    return trks


def parallel_track_dets_with_config(
    config: BYTETrackerConfig,
    dets_files: List[Path],
    n_proc: int = 1,
):
    """Batch track detections from a list of file paths with given parameters"""
    single_proc_func = functools.partial(track_dets_with_config, config)
    process_map(single_proc_func, dets_files, max_workers=n_proc)


parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, default="data/videos")
if __name__ == "__main__":
    args = parser.parse_args()

    # set paths
    model_path = Path(
        "/abyss/home/localcode/bb-yolo/"
        "models/models_20231219_ontology/"
        "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_"
        "20231219-054003/"
        "weights/best.pt"
    )

    video_dir = Path(args.video_dir)
    video_paths = list(video_dir.glob("*.[mM][pP]4"))

    # print file info
    print("Found video files:")
    print("\n".join([str(vp) for vp in video_paths]))

    # create detector config, "model" is required, other fields are optional
    detector_config = BBDetectorConfig(model=model_path)
    print(f"{detector_config=}")

    # initialize detector
    detector = BBDetector(config=detector_config)
    print(f"{detector=}")

    # iterate over video files
    dets_files = []
    for vp in video_paths:
        print(f"processing {vp=}")

        # get detections
        dets, dets_path = detector.detect(vp)
        dets_files.append(dets_path)

    # parallel tracking
    parallel_track_dets_with_config(
        config=BYTETrackerConfig(), dets_files=dets_files, n_proc=4
    )
