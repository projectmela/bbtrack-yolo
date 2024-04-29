from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig
from bbtrack_yolo.BBoxDetection import BBoxDetection

# load detections
dets = BBoxDetection.load_from("predictions/detection_190_5s.parquet")

# create tracker config
tracker_config = BYTETrackerConfig()

# create tracker
tracker = BBTracker(config=tracker_config)

# track detections
trks = tracker.track(dets)

# save tracks
trks.save_to("predictions/tracks_190_5s.csv")

# plot predictions on video
trks.plot_on("videos/0190_5s.mp4")
