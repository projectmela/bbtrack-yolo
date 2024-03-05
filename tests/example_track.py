from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig

dets = BBoxDetection.load_from("data/detection.csv")

tracker = BBTracker(config=BYTETrackerConfig())

trks = tracker.track(dets)

trks.plot_on("data/0190_0.5s.mp4")