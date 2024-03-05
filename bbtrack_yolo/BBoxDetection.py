""" Bounding box detection class for multiple frames """
from pathlib import Path
from typing import Union, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pandera as pa
from numpy import typing as npt
from tqdm.asyncio import tqdm


class BBoxDetectionSchema(pa.DataFrameModel):
    """ BBoxDetection Schema """
    file_path: str
    frame: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    confidence: float
    track_id: int
    class_id: float
    class_name: str


class BBoxDetection:
    """ Bounding box detection class for multiple frames

    Args:
        df (pd.DataFrame):
            DataFrame with detections, comply with BBoxDetectionSchema

    Attributes:
        _df (pd.DataFrame):
            DataFrame with detections, comply with BBoxDetectionSchema
        cls_id_to_name (Dict[int, str]):
            class_id to class_name mapping

    Methods:
        save_to(save_dir: Union[Path, str], csv: bool = False):
            save predictions as a parquet DataFrame file in the given directory
        load_from(file_path: Union[Path, str]) -> "BBoxDetection":
            load predictions from parquet or csv file
        save_to_mot17(file_path: Union[Path, str]):
            save to MOT17 format
        load_from_mot17(file_path: Union[Path, str],
                        class_id: int = -1,
                        class_name: str = "object") -> "BBoxDetection":
            load from MOT17 format txt file
        to_mot17() -> npt.NDArray:
            return predictions in MOT17 format
        plot_on(video_path: Union[Path, str],
                output_dir: Optional[Union[Path, str]] = None):
            plot boxes on video frames
        ltrb() -> npt.NDArray:
            return bboxes in ltrb (x1, y1, x2, y2) format
        xywh() -> npt.NDArray:
            return bboxes in xywh (center_x, center_y, width, height) format
        conf() -> npt.NDArray:
            return confidence
        cls_id() -> npt.NDArray:
            return class
        frame_range() -> Tuple[int, int]:
            return min and max frame number
        at(frame: int) -> "BBoxDetection":
            return detections at a specific frame
    """

    # TODO: upon pandera issue #763 fixed, update code
    # 1. type hint "df: pat.DataFrame[BBPredictionSchema]"
    # 2. remove validate method
    # 3. add decorators to the methods: @pa.check_types

    def __init__(
            self,
            df: pd.DataFrame
    ):
        BBoxDetectionSchema.validate(df)
        self._df = df.copy()

        # TODO: better way to handle class_id and class_name
        self.cls_id_to_name = (
            self._df.loc[["class_id", "class_name"]]
            .drop_duplicates()
            .set_index("class_id")["class_name"]
            .to_dict()
        )

    def save_to(self, save_dir: Union[Path, str], csv: bool = False):
        """ save predictions as a parquet DataFrame file in the given directory

        Args:
            save_dir (Union[Path, str]): directory to save the file
            csv (bool): save an extra csv file for human-readable format
        """

        save_dir = Path(save_dir)

        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        elif not save_dir.is_dir():
            raise ValueError("save_dir should be a directory")

        # not tracked if track_id all equal to -1, indicate in file name
        tracked = not self._df["track_id"].eq(-1).all()

        if tracked:
            file_name = "detection.parquet"
        else:
            file_name = "tracking.parquet"

        # save parquet file
        file_path = save_dir / file_name
        self._df.to_parquet(file_path)

        # save extra csv file if needed
        if csv:
            file_path_csv = file_path.with_suffix(".csv")
            self._df.to_csv(file_path_csv, index=False, header=True)

    @staticmethod
    def load_from(file_path: Union[Path, str]) -> "BBoxDetection":
        """ load predictions from a parquet or csv file

        Args:
            file_path (Union[Path, str]): file path to load from
        """
        file_path = Path(file_path)

        # check if is a parquet file
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")
        elif not file_path.is_file():
            raise ValueError("file_path should be a file")

        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError("file_path should be a parquet or csv file")

        return BBoxDetection(df=df)

    def save_to_mot17(
            self,
            file_path: Union[Path, str]
    ):
        """ save to MOT17 format """

        file_path = Path(file_path)

        if file_path.exists():
            raise FileExistsError(f"{file_path} already exists")
        elif file_path.suffix != ".txt":
            raise ValueError("file_path should be a txt file")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.to_mot17().tofile(file_path, sep=",", format="%s")

    @staticmethod
    def load_from_mot17(
            file_path: Union[Path, str],
            class_id: int = -1,
            class_name: str = "object"
    ) -> "BBoxDetection":
        """ load from MOT17 format txt file """

        file_path = Path(file_path)

        # check if is a parquet file
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")
        elif not file_path.is_file():
            raise ValueError("file_path should be a file")
        elif file_path.suffix != ".txt":
            raise ValueError("file_path should be a txt file")

        # read txt file
        df = pd.read_csv(file_path, header=None,
                         names=["frame", "track_id",
                                "bb_left", "bb_top", "bb_width", "bb_height",
                                "confidence", "x", "y", "z"])
        # convert to dtype
        df = df.astype({
            "frame": int,
            "track_id": int,
            "bb_left": float,
            "bb_top": float,
            "bb_width": float,
            "bb_height": float,
            "confidence": float,
            "x": int,
            "y": int,
            "z": int
        })

        # add class_id and class_name
        df["class_id"] = class_id
        df["class_name"] = class_name

        df = df.loc[["frame", "bb_left", "bb_top", "bb_width", "bb_height",
                     "confidence", "track_id", "class_id", "class_name"]]

        return BBoxDetection(df)

    def to_mot17(self) -> npt.NDArray:
        """ return predictions in MOT17 format """
        mot_df = self._df.copy()
        mot_df["x"] = -1
        mot_df["y"] = -1
        mot_df["z"] = -1
        mot_df = mot_df.loc[["frame", "track_id",
                             "bb_left", "bb_top", "bb_width", "bb_height",
                             "confidence", "x", "y", "z"]]
        return mot_df.to_numpy()

    def plot_on(
            self,
            video_path: Union[Path, str],
            output_dir: Optional[Union[Path, str]] = None
    ):
        """plot boxes on video frames"""
        video_path = Path(video_path)
        output_dir = Path(output_dir) if output_dir else None
        if output_dir is None:
            output_dir = video_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{video_path.stem}_plotted.mp4"

        src_vc = cv2.VideoCapture(str(video_path))
        fps = src_vc.get(cv2.CAP_PROP_FPS)
        frame_width = int(src_vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(src_vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out_vc = cv2.VideoWriter(str(output_file), fourcc, fps,
                                 (frame_width, frame_height))

        # check if frames are equal
        n_video_frames = int(src_vc.get(cv2.CAP_PROP_FRAME_COUNT))
        n_pred_frames = self._df["frame"].nunique()
        if n_video_frames != n_pred_frames:
            tqdm.write(f"Warning: video frames {n_video_frames}, "
                       f"prediction frames {n_pred_frames}")

        # create random colors for each object
        n_objects = 200
        np.random.seed(42)
        colors = np.random.rand(n_objects, 3)
        # convert to tuple(int) and de-normalize
        color_map = [tuple(map(int, c * 255)) for c in colors]

        # other format
        font_scale = 1.3
        font_thickness = 3
        line_width = 2

        # iterate over frames
        for frame_id, frame in tqdm(enumerate(range(n_video_frames)),
                                    total=n_video_frames,
                                    bar_format="{l_bar}{bar:10}{r_bar}"):
            ret, img = src_vc.read()
            if not ret:
                break

            # plot boxes
            for _, row in self._df[self._df["frame"].eq(frame_id + 1)].iterrows():
                track_id = row["track_id"]
                color = color_map[track_id % n_objects]

                x, y, w, h = (
                    row[["bb_left", "bb_top", "bb_width", "bb_height"]].astype(int)
                )

                # TODO: add class_name and confidence
                cv2.rectangle(img, (x, y), (x + w, y + h), color, line_width)
                # plot id
                shift = 8
                cv2.putText(
                    img,
                    str(track_id),
                    (x - shift, y - shift),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thickness,
                )

            # write frame
            out_vc.write(img)

        # release video capture and video writer
        src_vc.release()
        out_vc.release()

        # check if the output file exists
        if not output_file.exists():
            raise FileNotFoundError(f"Failed to create {output_file}")

    @property
    def ltrb(self) -> npt.NDArray:
        """ return bboxes in ltrb (x1, y1, x2, y2) format """
        ltrb = self._df.loc[["bb_left", "bb_top", "bb_width", "bb_height"]].copy()
        ltrb["bb_width"] += ltrb["bb_left"]
        ltrb["bb_height"] += ltrb["bb_top"]
        return ltrb.to_numpy()

    @property
    def xywh(self) -> npt.NDArray:
        """ return bboxes in xywh (center_x, center_y, width, height) format """
        xywh = self._df.loc[["bb_left", "bb_top", "bb_width", "bb_height"]].copy()
        xywh["bb_left"] += xywh["bb_width"] / 2
        xywh["bb_top"] += xywh["bb_height"] / 2
        return xywh.to_numpy()

    @property
    def conf(self) -> npt.NDArray:
        """ return confidence """
        return self._df["confidence"].to_numpy()

    @property
    def cls_id(self) -> npt.NDArray:
        """ return class """
        return self._df["class_id"].to_numpy()

    @property
    def frame_range(self) -> Tuple[int, int]:
        """ return min and max frame number """
        return self._df["frame"].min(), self._df["frame"].max()

    def at(self, frame: int) -> "BBoxDetection":
        """ return detections at a specific frame """
        return BBoxDetection(self._df[self._df["frame"].eq(frame)])
