""" A tracker class to track bbox detections with YOLOv8-implemented BYTETracker """

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic.dataclasses import dataclass
from tqdm.auto import tqdm
from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore

from bbtrack_yolo.BBoxDetection import BBoxDetection


@dataclass(frozen=True)
class BYTETrackerConfig:
    """Arguments for BYTETracker implementation in YOLOv8"""

    track_high_thresh: float = 0.1  # threshold for the first association
    track_low_thresh: float = 0.01  # threshold for the second association
    new_track_thresh: float = 0.1  # threshold for init new track if the detection does
    # not match any tracks

    track_buffer: int = 30  # buffer to calculate the time when to remove tracks
    match_thresh: float = 0.8  # threshold for matching tracks

    # threshold for min box areas (for tracker evaluation, not used for now)
    # min_box_area: 10
    # mot20: False # for tracker evaluation(not used for now)

    def __str__(self):
        return (
            f"hi={self.track_high_thresh}_"
            f"lo={self.track_low_thresh}_"
            f"new={self.new_track_thresh}_"
            f"buf={self.track_buffer}_"
            f"match={self.match_thresh}"
        )


class Config:
    """Pydantic config to allow arbitrary types such as numpy arrays"""

    arbitrary_types_allowed = True


@dataclass(config=Config)
class BYTEDetection:
    """Detections in a single frame
    BYTETracker implementation in YOLOv8 expect the following format of detections
    """

    xywh: npt.NDArray  # shape (N, 4), bbox in (center x, center y, width, height)
    conf: npt.NDArray  # shape (N, ), confidence of the detections
    cls: npt.NDArray  # shape (N, ), class of the detections


class BBTracker:
    """A tracker implemented to track the bounding boxes of the detected objects

    Args:
        config: BYTETrackerConfig, arguments for BYTETracker implementation

    Attributes:
        config(BYTETrackerConfig): arguments for BYTETracker implementation
        tracker(BYTETracker): BYTETracker implementation in YOLOv8

    Methods:
        track(dets: BBoxDetection) -> BBoxDetection: Track the bounding boxes
    """

    def __init__(self, config: BYTETrackerConfig):
        self.config = config
        self.tracker = BYTETracker(
            args=config, frame_rate=30  # TODO: get frame rate from video
        )

    def track(self, dets: BBoxDetection, reset: bool = True) -> BBoxDetection:
        """Track the bounding boxes"""

        if reset:
            self.tracker.reset()

        tracking_results = np.empty((0, 9))
        # convert BBPrediction to BYTEDetection
        for frame_idx in tqdm(
            range(dets.frame_range[0], dets.frame_range[1] + 1),
            bar_format="{l_bar}{bar:10}{r_bar}",
            desc="Tracking",
        ):
            # get detections at the current frame
            frame_dets = dets.at(frame_idx)

            # update tracker and obtain tracks
            # (shape (N, 6), (x1, y1, x2, y2, track_id, conf, cls, det_idx_in_frame))
            tracks = self.tracker.update(
                results=BYTEDetection(
                    xywh=frame_dets.xywh, conf=frame_dets.conf, cls=frame_dets.cls_id
                )
            )

            if tracks.size == 0:
                continue

            # concatenate the frame index to the tracks
            tracks = np.concatenate(
                [np.full((tracks.shape[0], 1), frame_idx), tracks], axis=1
            )

            # append the tracks to the tracking results
            tracking_results = np.concatenate([tracking_results, tracks], axis=0)

        # convert from ltrb (x1, y1, x2, y2) to ltwh (left, top, width, height)
        tracking_results[:, 3] -= tracking_results[:, 1]
        tracking_results[:, 4] -= tracking_results[:, 2]

        df = pd.DataFrame(
            tracking_results,
            columns=[
                "frame",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "track_id",
                "confidence",
                "class_id",
                "det_idx_in_frame",
            ],
        )

        # remove det_idx_in_frame
        df = df.drop(columns=["det_idx_in_frame"])

        # convert track_id, frame to int
        df["frame"] = df["frame"].astype(int)
        df["track_id"] = df["track_id"].astype(int)

        # add dummy file_path
        # TODO: what to do with file_path?
        df["file_path"] = ""

        # add class_name
        df["class_name"] = df["class_id"].apply(lambda x: dets.cls_id_to_name[x])

        return BBoxDetection(df)
