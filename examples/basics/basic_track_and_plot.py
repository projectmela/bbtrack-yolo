import torch
from boxmot import OCSORT, BoTSORT, BYTETracker
from pathlib import Path
from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBoxTracker import BBoxTracker


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


# load detections
all_dets = BBoxDetection.load_from("examples/basics/predictions/detection_190_5s.parquet")

# video path, if re_id is given video must be provided 
video_path = "examples/basics/videos/0190_5s.mp4"

# FOR Custom videos 
# all_dets = BBoxDetection.load_from("PATH/TO/DETECTION_FILE.parquet/.csv")
# video_path = "PATH/TO/VIDEO.MP4"

# initialize tracker

# Example 1 : OC Sort
# tracker = BBoxTracker(tracker=OCSORT())

# Example 2 : Byte tracker 
# tracker = BBoxTracker(BYTETracker())

# Example 3 : BotSort without any specific weight, without Re-ID
# tracker = BBoxTracker(
    #     BoTSORT(
    #         model_weights=None,
    #         device=get_device(),
    #         fp16=False,
    #         with_reid=False,
    #     ),
# #     )

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

# run tracking for detections
trks = tracker.track(all_dets, video_path= video_path)

# save tracks
#trks.save_to("PATH/TO/CUSTOM_NAME.csv")
trks.save_to("examples/basics/predictions/track_190_5s.csv")

# plot predictions on video
# trks.plot_on("PATH/TO/TARGET_VIDEO_FILE.MP4") or define above 
trks.plot_on(video_path)
