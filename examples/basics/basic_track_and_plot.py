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
all_dets = BBoxDetection.load_from("examples/basics/predictions/detection.csv")

#all_dets = BBoxDetection.load_from("predictions/detection_190_5s.parquet")

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
# # )

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


# track detections
trks = tracker.track(all_dets)

# save tracks
trks.save_to("examples/basics/predictions/20230313_SE_Lek1_P1D1_DJI_0295.csv")

# plot predictions on video
# trks.plot_on("/home/hnaik/mela_yolo/bbtrack-yolo/dataset/test/20230313_SE_Lek1_P1D1_DJI_0295.MP4")

#trks.plot_on("examples/basics/videos/0190_5s.mp4")
