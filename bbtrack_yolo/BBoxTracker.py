from typing import Union

import numpy as np
import pandas as pd
from boxmot import BYTETracker, OCSORT, BoTSORT, HybridSORT, StrongSORT
from tqdm.auto import trange

from bbtrack_yolo.BBoxDetection import BBoxDetection


class BBoxTracker:
    """A wrapper class built on top of trackers from BoxMOT implementation"""

    def __init__(
        self,
        tracker: Union[BYTETracker, OCSORT, BoTSORT, HybridSORT, StrongSORT],
        frame_width: int = 5472,
        frame_height: int = 3078,
    ):
        """Initialize the tracker"""

        self._tracker = tracker
        self._empty_img = np.empty((frame_height, frame_width, 3), dtype=np.uint8)

    def __str__(self):
        tracker_cls_name = self._tracker.__class__.__name__
        if tracker_cls_name == "BYTETracker":
            tracker_name = "BYTE"
            # trker = self._tracker
            # param_str = (
            #     f"{tracker_name}"
            #     f"-trk_th={trker.track_thresh}"
            #     f"-match_th={trker.match_thresh}"
            #     f"-buf={trker.track_buffer}"
            # )
            return tracker_name
        elif tracker_cls_name == "OCSORT":
            tracker_name = "OCST"
            # trker = self._tracker
            # param_str = (
            #     f"{tracker_name}"
            #     f"-det_thresh={trker.det_thresh}"
            #     f"-max_age={trker.max_age}"
            #     f"-min_hits={trker.min_hits}"
            #     f"-asso_th={trker.asso_threshold}"
            #     f"-delta_t={trker.delta_t}"
            #     f"-inertia={trker.inertia}"
            #     f"-use_bytes={trker.use_byte}"
            # )
            return tracker_name
        elif tracker_cls_name == "BoTSORT":
            tracker_name = "BoTST"
            # trker = self._tracker
            # param_str = (
            #     f"{tracker_name}"
            #     f"-h={trker.track_high_thresh}"
            #     f"-l={trker.track_low_thresh}"
            #     f"-new={trker.new_track_thresh}"
            #     f"-mat={trker.match_thresh}"
            #     f"-buf={trker.track_buffer}"
            #     f"-prox={trker.proximity_thresh}"
            #     f"-appear={trker.appearance_thresh}"
            # )
            return tracker_name
        elif tracker_cls_name == "HybridSORT":
            tracker_name = "HyST"
            return tracker_name
        elif tracker_cls_name == "StrongSORT":
            tracker_name = "StST"
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

            # Update tracker and get tracks with dets: (x, y, x, y, conf, cls)
            tracks = self._tracker.update(
                dets=dets,
                # Use a dummy empty image since we don't need plot
                # but have to pass frame dimension information
                img=self._empty_img,
            )  # => (n, 8) as (x, y, x, y, id, conf, cls, ind)

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
