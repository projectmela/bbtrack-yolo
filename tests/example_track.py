import matplotlib.pyplot as plt

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBTracker import BBTracker, BYTETrackerConfig

dets = BBoxDetection.load_from("data/detection.csv")

# display the confidence histogram
dets.confidence_histogram()
# save the image
plt.savefig("data/confidence_histogram.png")

tracker_config = BYTETrackerConfig(
    # threshold for the first association
    track_high_thresh=0.5,
    # threshold for the second association
    track_low_thresh=0.1,
    # threshold for init new track if the detection does not match any tracks
    new_track_thresh=0.1,
    # buffer to calculate the time when to remove tracks
    track_buffer=30,
    # threshold for matching tracks
    match_thresh=0.8,
)

tracker = BBTracker(config=BYTETrackerConfig())

trks = tracker.track(dets)

trks.plot_on("data/0190_0.5s.mp4")
