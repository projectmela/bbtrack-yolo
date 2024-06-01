from typing import Union, TypeVar

import numpy as np
import pandas as pd
from boxmot import BYTETracker, OCSORT, BoTSORT, HybridSORT, StrongSORT
from tqdm.auto import trange

from bbtrack_yolo.BBoxDetection import BBoxDetection

TrackerType = TypeVar(
    "TrackerType", BYTETracker, OCSORT, BoTSORT, HybridSORT, StrongSORT
)


class BBoxTracker:
    """A wrapper class built on top of trackers from BoxMOT implementation"""

    def __init__(
        self,
        tracker: TrackerType,
        frame_width: int = 5472,
        frame_height: int = 3078,
        reid_model_name: str = "",
    ):
        """Initialize the tracker"""

        self._tracker = tracker
        self._empty_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self._reid_model_name = reid_model_name

    def __str__(self):
        tracker_cls_name = self._tracker.__class__.__name__
        if tracker_cls_name == "BYTETracker":
            tracker_name = "BYTE"
            return tracker_name
        elif tracker_cls_name == "OCSORT":
            tracker_name = "OCST"
            return tracker_name
        elif tracker_cls_name == "BoTSORT":
            tracker_name = "BoTST"
            trker = self._tracker
            if trker.with_reid:
                tracker_name += f"_reid={self._reid_model_name}"
            return tracker_name
        elif tracker_cls_name == "HybridSORT":
            tracker_name = "HyST"
            tracker_name += f"_reid={self._reid_model_name}"
            return tracker_name
        elif tracker_cls_name == "StrongSORT":
            tracker_name = "StST"
            tracker_name += f"_reid={self._reid_model_name}"
            return tracker_name
        else:
            tracker_name = tracker_cls_name
            return tracker_name

    def track(
        self,
        all_dets: BBoxDetection,
        progress_bar: bool = True,
    ) -> BBoxDetection:
        """Track the bounding boxes
        :param all_dets: BBoxDetection object containing detections from all frames
        :param progress_bar: show progress bar
        :return: BBoxDetection object containing tracking results
        """

        tracking_results = np.empty((0, 9))
        for frame_idx in trange(
            all_dets.max_frame + 1,
            bar_format="{l_bar}{bar:10}{r_bar}",
            desc="Tracking",
            disable=(not progress_bar),
        ):
            # Get detections at the current frame
            dets = all_dets.at(frame=frame_idx).ltrb_conf_clsid

            try:
                # Update tracker and get tracks with dets: (x, y, x, y, conf, cls)
                tracks = self._tracker.update(
                    dets,
                    # Use a dummy empty image since we don't need plot
                    # but have to pass frame dimension information
                    self._empty_img,
                )  # => (n, 8) as (x, y, x, y, id, conf, cls, ind)
            except Exception as e:
                print(f"Error at frame {frame_idx}: {e}")
                # convert dets to df
                df = pd.DataFrame(
                    dets,
                    columns=[
                        "bb_left",
                        "bb_top",
                        "bb_right",
                        "bb_bottom",
                        "confidence",
                        "class_id",
                    ],
                )
                print(f"\n{df.to_markdown()}\n")
                raise e

            if tracks.size == 0:
                continue
            else:
                # Add frame index to the tracks
                tracks = np.concatenate(
                    [np.full((tracks.shape[0], 1), frame_idx), tracks],
                    axis=1,
                )
                # Append the tracks to the tracking results
                tracking_results = np.concatenate(
                    [tracking_results, tracks],
                    axis=0,
                )

        # Convert from ltrb (x1, y1, x2, y2) to ltwh (left, top, width, height)
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
        df["class_name"] = df["class_id"].apply(lambda x: all_dets.cls_id_to_name[x])

        return BBoxDetection(df)
