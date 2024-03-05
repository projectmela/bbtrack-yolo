from pathlib import Path
from typing import Union, Optional

import cv2
import pandas as pd
import pandera as pa
from numpy import typing as npt
from tqdm.asyncio import tqdm


class BBPredictionSchema(pa.DataFrameModel):
    """ schema for BBPrediction """
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


class BBPrediction:
    """ bounding box prediction class """

    # TODO: upon pandera issue #763 fixed, update code
    # 1. type hint "df: pat.DataFrame[BBPredictionSchema]"
    # 2. remove validate method
    # 3. add decorators to the methods: @pa.check_types

    def __init__(
            self,
            df: pd.DataFrame
    ):
        self.seq_name = ""
        self.frame_width: int
        self.frame_height: int

        BBPredictionSchema.validate(df)
        self._df = df.copy()

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
            file_name = f"{self.seq_name}_tracked.parquet"
        else:
            file_name = f"{self.seq_name}_untracked.parquet"

        # save parquet file
        file_path = save_dir / file_name
        self._df.to_parquet(file_path)

        # save extra csv file if needed
        if csv:
            file_path_csv = file_path.with_suffix(".csv")
            self._df.to_csv(file_path_csv, index=False, header=True)

    @staticmethod
    def load_from(file_path: Union[Path, str]) -> "BBPrediction":
        """ load predictions from a parquet file

        Args:
            file_path (Union[Path, str]): file path to load from
        """
        file_path = Path(file_path)

        # check if is a parquet file
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")
        elif not file_path.is_file():
            raise ValueError("file_path should be a file")
        elif file_path.suffix != ".parquet":
            raise ValueError("file_path should be a parquet file")

        df = pd.read_parquet(file_path)

        return BBPrediction(df=df)

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
    ) -> "BBPrediction":
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

        df = df[["frame", "bb_left", "bb_top", "bb_width", "bb_height",
                 "confidence", "track_id", "class_id", "class_name"]]

        return BBPrediction(df)

    def to_mot17(self) -> npt.NDArray:
        """ return predictions in MOT17 format """
        mot_df = self._df.copy()
        mot_df["x"] = -1
        mot_df["y"] = -1
        mot_df["z"] = -1
        mot_df = mot_df[["frame", "track_id",
                         "bb_left", "bb_top", "bb_width", "bb_height",
                         "confidence", "x", "y", "z"]]
        return mot_df.to_numpy()

    def plot_on(
            self,
            video_path: Union[Path, str],
            output_dir: Optional[Union[Path, str]] = None
    ):
        """plot boxes on video frames"""

        if output_dir is None:
            output_dir = Path(video_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.seq_name}_plotted.mp4"

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

        # iterate over frames
        for frame_id, frame in tqdm(enumerate(range(n_video_frames)),
                                    total=n_video_frames,
                                    bar_format="{l_bar}{bar:10}{r_bar}"):
            ret, img = src_vc.read()
            if not ret:
                break

            # plot boxes
            for _, row in self._df[self._df["frame"].eq(frame_id + 1)].iterrows():
                x, y, w, h = (
                    row[["bb_left", "bb_top", "bb_width", "bb_height"]].astype(int)
                )
                # TODO: add track_id, color, class_name and confidence
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # write frame
            out_vc.write(img)

        # release video capture and video writer
        src_vc.release()
        out_vc.release()

        # check if the output file exists
        if not output_file.exists():
            raise FileNotFoundError(f"Failed to create {output_file}")
