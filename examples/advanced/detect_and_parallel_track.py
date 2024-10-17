import argparse
import functools
from pathlib import Path
from typing import List
import torch
from tqdm.contrib.concurrent import process_map

from bbtrack_yolo.BBDetecor import BBDetector, BBDetectorConfig
from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig, BotSortTrackerConfig
from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBoxTracker import BoTSORT, BBoxTracker

# Get device information from the computer about GPU, CPU etc. 
def get_device():
    """Automatically select device for torch based on environment / hardware"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        # Default fallback to CPU
        return "cpu"

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
parser.add_argument("--video_dir", type=str, default="examples/basics/videos")
parser.add_argument("--model_path", type=str, default="examples/basics/nano_bb.pt")
# TO allow for multi-threading 
parser.add_argument("--thread", type=bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    video_dir = Path(args.video_dir)
    model_path = Path(args.model_path)
    video_paths = list(video_dir.glob("*.[mM][pP]4"))

    print("Video Path : ", video_dir)
    print("Model Path : ", model_path)

    # print video files info
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

    if (args.thread):
        # parallel tracking
        parallel_track_dets_with_config(
            config=BYTETrackerConfig(), dets_files=dets_files, n_proc=4
        )
    else:            
        # Singular tracking function 
        tracker = BBoxTracker(
            BoTSORT(
                model_weights= Path("osnet_ain_x1_0_msmt17.pt"), #"osnet_x0_25_msmt17.pt", #"resnet50_msmt17",
                # more re-id models see:
                # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
                device=get_device(),
                fp16=False,
                with_reid=True,
            ),
            reid_model_name= "osnet_ain_x1_0_msmt17.pt",
        )

        # load files for tracking 
        for file in dets_files:
            dets = BBoxDetection.load_from(file)

                    # track detections
            trks = tracker.track(dets)

            # save tracks
            trks.save_to(file.parent / "tracks.csv")
            
    print("All files processed")