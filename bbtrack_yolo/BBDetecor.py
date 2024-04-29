""" Bounding Box Detector class """

import json
import os
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union

import torch
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass
from tqdm.auto import tqdm
from ultralytics import YOLO  # type: ignore

from bbtrack_yolo.BBoxDetection import BBoxDetection
from utility import cur_dt_str


@dataclass(frozen=True)
class BBDetectorConfig:
    """
    Bounding Box Detector Arguments
    """

    model: Union[str, Path]  # model name or model path

    # training parameters
    data: Optional[str] = None  # path to the dataset yaml file
    ontology: str = ""  # ontology code name of dataset

    imgsz: int = Field(gt=0, default=1280)  # image size for training
    batch: int = Field(gt=0, default=1)
    epochs: int = Field(gt=0, default=10)  # maximum number of epochs
    patience: int = Field(gt=0, default=10)  # early stopping patience
    save_period: Optional[int] = Field(gt=0, default=None)  # save model snapshots gap

    workers: int = Field(gt=0, default=2)  # number of workers for dataloader
    resume: bool = False  # resume training

    # training parameters that often not changed
    seed: int = 0  # random seed
    deterministic: bool = True  # reproducible
    optimizer: str = "auto"  # optimizer
    # device: str = "0"  # device to use
    project = "models"  # directory to save the trained model

    # prediction parameters
    pred_save_dir: str = "predictions"  # directory to save the predictions
    pred_th: float = Field(ge=0.0, default=0.0)  # pred threshold for confidence (score)

    # save predictions with some classes only
    keep_only_classes: List[str] = Field(
        default_factory=lambda: ["blackbuck", "bbmale", "bbfemale", "bb"]
    )

    @classmethod
    @field_validator("model", mode="before")
    def model_path_to_str(cls, v):
        """convert to str when model is a Path"""
        return v.as_posix() if isinstance(v, Path) else v

    @cached_property
    def name(self) -> str:
        """name of the model, only calculated once, thus datetime is fixed"""
        return f"{self.model_name}_{cur_dt_str()}"

    @property
    def device(self) -> str:
        """device to use, if mps is available, use it"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "0"
        else:
            return "cpu"

    @property
    def model_name(self) -> str:
        """model name for training or inference naming"""
        if YOLO.is_hub_model(str(self.model)) and self.data is not None:
            # a base model from ultralytics, generate model_name in training case
            return (
                f"d={Path(self.data).stem}_"
                f"o={self.ontology}_"
                f"{self.model_train_param_str}"
            )
        elif (
            Path(self.model).exists()
            and Path(self.model).is_file()
            and Path(self.model).suffix == ".pt"
            and Path(self.model).parent.name == "weights"
        ):
            # a model path, generate model_name in inference case
            return Path(self.model).parent.parent.name
        else:
            raise ValueError(f"Unable to give model_name for current config {self=}.")

    @property
    def model_train_param_str(self) -> str:
        """model training parameters as string"""
        return f"m={Path(self.model).stem}_" f"imgsz={self.imgsz}_" f"bs={self.batch}"


class BBDetector:
    """Bounding Box Detector

    Attributes:
        config(BBDetectorConfig): configuration of the detector
        model(ultralytics.YOLO): the model to detect bounding boxes

    Methods:
        train():
            not implemented yet
        eval():
            not implemented yet
        detect(source: Union[str, Path]) -> BBoxDetection:
            detect bounding boxes from source
    """

    config: BBDetectorConfig

    def __init__(
        self,
        config: BBDetectorConfig,
    ):
        self.config = config
        self.model = YOLO(config.model)

    def train(self):
        """train the model with configs"""
        raise NotImplementedError("Training is not implemented yet.")
        # try to log training with comet
        COMET_API_KEY = os.getenv("COMET_API_KEY")
        COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
        COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
        try:
            from comet_ml import Experiment  # type: ignore

            experiment = Experiment(
                api_key=COMET_API_KEY,
                project_name=COMET_PROJECT_NAME,
                workspace=COMET_WORKSPACE,
            )
            # add date as tag
            experiment.add_tag(f'train_{cur_dt_str().split("-", 1)[0]}')
        except Exception as e:
            print(
                f"model training not logged since comet failed to initialize.\n"
                f"{COMET_API_KEY=}, {COMET_PROJECT_NAME=}, {COMET_WORKSPACE=}\n"
                f"{e}"
            )
            # TODO: give detail why comet failed to initialize, look for .comet.config
            experiment = None

        # set model name and experiment name
        if experiment is not None:
            experiment.set_name(self.config.name)

        # train model
        self.model.train(**asdict(self.config))

    def eval(self):
        """evaluate the model"""
        raise NotImplementedError("Evaluation is not implemented yet.")
        metrics = self.model.val(
            # data=, # need to specify dataset.yaml if not in default location
            imgsz=self.config.imgsz,
            batch=1,
            conf=0.001,  # default
            iou=0.6,  # default
            max_det=500,  # default
            device=self.config.device,
            # save to project/name
            project=f"models/{self.config.name}",
            name="validation",
        )  # returns: ultralytics.utils.metrics.Metric

        print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
        cls_map50_95 = metrics.box.maps
        cls_ap50 = metrics.box.ap50
        model_eval_results = {
            "model_name": self.config.name,
            "datetime": cur_dt_str(),
            "ontology": self.config.ontology,
            "cls_names": ",".join(self.model.names.values()),
            "model": self.config.model,
            "dataset": self.config.data,
            "image_size": self.config.imgsz,
            "batch_size": self.config.batch,
        }
        model_eval_results.update(
            {
                f"{self.model.names[cls]}_map": _map
                for cls, _map in zip(self.model.names, cls_map50_95)
            }
        )
        model_eval_results.update(
            {
                f"{self.model.names[cls]}_ap50": ap50
                for cls, ap50 in zip(self.model.names, cls_ap50)
            }
        )
        print(f"{model_eval_results=}")

        # dump as json
        with open(f"models/{self.config.name}/processed_eval.json", "w") as f:
            json.dump(model_eval_results, f)

    def detect(self, source: Union[str, Path]) -> (BBoxDetection, Path):
        """detect bounding boxes from source"""
        model_name = self.config.model_name
        source_name = Path(source).stem
        pred_save_dir = Path(self.config.pred_save_dir) / model_name
        pred_name = f"{cur_dt_str()}_{source_name}"
        results = self.model.predict(
            source,
            # path parameters
            project=pred_save_dir.as_posix(),
            name=pred_name,
            # inference parameters
            stream=True,  # avoid memory overflow
            device=self.config.device,
            conf=self.config.pred_th,
            max_det=500,
            # output parameters
            save_conf=True,
        )  # in stream mode, return a generator

        if self.config.keep_only_classes:
            tqdm.write(f"Predictions saved with {self.config.keep_only_classes} only.")

        csv_path = self._stream_write_predictions(
            results, save_dir=pred_save_dir / pred_name
        )

        # convert the written result from csv to BBPrediction
        return BBoxDetection.load_from(csv_path), csv_path

    def _stream_write_predictions(
        self,
        results,  # ultralytics.engine.results.Results
        save_dir: Union[str, Path],
    ) -> Path:
        """
        write results to csv in streamline in case of error and return BBPrediction
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # create scv file
        csv_path = save_dir / "detection.csv"
        csv_file = open(csv_path, "w")
        # write csv header
        csv_file.write(
            "file_path,frame,track_id,"
            "bb_left,bb_top,bb_width,bb_height,"
            "confidence,class_id,class_name\n"
        )

        # collect results into human readable dataframes
        frame_start = 0  # opencv video frame id starts from 0, mot from 1
        frame_id = frame_start
        last_path = None
        try:
            for result in results:
                result = result.cpu()  # move result to cpu

                # set frame_id according to file_path
                if result.path == last_path:
                    # different image from same video, increment id
                    frame_id += 1
                else:
                    # a new video or image, reset id
                    frame_id = frame_start
                    last_path = result.path

                # save box detections to csv
                for box in result.numpy().boxes:
                    bbox = box.xywh.reshape(-1)
                    bbl, bbt = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
                    bbw, bbh = bbox[2], bbox[3]
                    conf, cls = box.conf.item(), box.cls.item()
                    cls_name = (
                        self.model.names[cls]
                        if self.model.names is not None
                        else str(cls)
                    )
                    if (
                        self.config.keep_only_classes
                        and cls_name not in self.config.keep_only_classes
                    ):
                        continue
                    csv_file.write(
                        f'"{result.path}",'
                        f"{frame_id},"
                        f"-1,"  # dummy track id
                        f"{bbl},{bbt},{bbw},{bbh},"
                        f"{conf},{cls},{cls_name}\n"
                    )
        finally:
            csv_file.close()
            return csv_path
