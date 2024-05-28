from boxmot import OCSORT

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBoxTracker import BBoxTracker

# load detections
all_dets = BBoxDetection.load_from("predictions/detection_190_5s.parquet")

# initialize tracker
tracker = BBoxTracker(tracker=OCSORT())

# track detections
trks = tracker.track(all_dets)

# save tracks
# trks.save_to("predictions/tracks_190_5s.csv")

# plot predictions on video
trks.plot_on("videos/0190_5s.mp4")
