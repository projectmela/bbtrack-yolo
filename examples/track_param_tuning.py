import itertools
from typing import Iterable

from bbtrack_yolo.BBoxDetection import BBoxDetection
from bbtrack_yolo.BBTracker import BYTETrackerConfig, BBTracker


# @dataclass(frozen=True)
# class BYTETrackerConfig:
#     """Arguments for BYTETracker implementation in YOLOv8"""
#
#     track_high_thresh: float = 0.1  # threshold for the first association
#     track_low_thresh: float = 0.01  # threshold for the second association
#     new_track_thresh: float = 0.1  # threshold for init new track if the detection does
#     # not match any tracks
#
#     track_buffer: int = 30  # buffer to calculate the time when to remove tracks
#     match_thresh: float = 0.8  # threshold for matching tracks
#
#     # threshold for min box areas (for tracker evaluation, not used for now)
#     # min_box_area: 10
#     # mot20: False # for tracker evaluation(not used for now)


# tracking param options
def tracking_params() -> Iterable[BYTETrackerConfig]:
    """Generate tracking parameters for tuning"""

    param_options = {
        "track_high_thresh": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "track_low_thresh": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "new_track_thresh": [0.1, 0.2, 0.3, 0.4, 0.5],
        "track_buffer": [10, 20, 30, 40, 50, 60],
        "match_thresh": [0.5, 0.6, 0.7, 0.8, 0.9],
    }

    for values in itertools.product(*param_options.values()):  # type: ignore
        params = dict(zip(param_options.keys(), values))
        yield BYTETrackerConfig(**params)


print(f"tracking_params: {next(tracking_params())}")

exit(0)

# load detections
dets = BBoxDetection.load_from("../tests/data/detection.csv")

# tracking
for params in tracking_params():
    tracker = BBTracker(config=params)
    trks = tracker.track(dets)

    # evaluate tracking

    # merge evaluations with tracking parameters

    # save merged evaluation

# evaluate tracking

# merge evaluations with tracking parameters

# save merged evaluation
