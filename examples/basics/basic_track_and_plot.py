import torch
from boxmot import OCSORT, BoTSORT, BYTETracker

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
all_dets = BBoxDetection.load_from("predictions/detection_190_5s.parquet")

# initialize tracker
# tracker = BBoxTracker(tracker=OCSORT())
# BBoxTracker(BYTETracker())
# BBoxTracker(
#     BoTSORT(
#         model_weights=None,
#         device=get_device(),
#         fp16=False,
#         with_reid=False,
#     ),
# )
tracker = BBoxTracker(
    BoTSORT(
        model_weights="resnet50_msmt17.pt",
        # more re-id models see:
        # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
        device=get_device(),
        fp16=False,
        with_reid=False,
    ),
 )

# track detections
trks = tracker.track(all_dets)

# save tracks
trks.save_to("predictions/tracks_190_5s.csv")

# plot predictions on video
trks.plot_on("videos/0190_5s.mp4")
