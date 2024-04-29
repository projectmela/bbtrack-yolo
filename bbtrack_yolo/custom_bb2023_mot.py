"""Custom configuration for the evaluation of the BB2023 dataset using TrackEval"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import trackeval  # type: ignore
from trackeval.datasets import MotChallenge2DBox  # type: ignore


def cur_dt_str():
    """get current datetime in string format
    :return: current datetime in string format "yyyymmdd-hhmmss"
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class CustomBB2023MOT(MotChallenge2DBox):
    """
    A custom class for the BB2023 dataset using TrackEval

    This class inherits MotChallenge2DBox dataset class from the TrackEval library, and
    further integrates the evaluation process for the BB2023 dataset.

    Attributes:
        config (Dict[str, Any]):
            A dictionary containing the dataset configuration, stored in superclass.
        eval_config (Dict[str, Any]):
            A dictionary containing the evaluation configuration for the dataset.
        metrics_config (Dict[str, Any]):
            A dictionary containing the metrics configuration for the dataset.
        evaluator (trackeval.Evaluator):
            An evaluator object to evaluate the tracker performance on the dataset.

    Methods:
        evaluate() -> Dict[str, Dict[str, Dict[str, Any]]]:
            Evaluate the tracker performance on the dataset and return the results.
        set_eval_config(eval_config: Dict[str, Any]):
            Set the evaluation configuration with specified values.
        set_metrics_config(metrics_config: Dict[str, Any]):
            Set the metrics configuration with specified values.
        set_dataset_config(dataset_config: Dict[str, Any]):
            Set the dataset configuration with specified values.
        get_default_dataset_config() -> Dict[str, Any]:
            Return the default dataset configuration for the BB2023 dataset.
        get_default_eval_config() -> Dict[str, Any]:
            Return the default evaluation configuration for the BB2023 dataset.
        get_default_metrics_config() -> Dict[str, Any]:
            Return the default metrics configuration for the BB2023 dataset.
        get_default_metrics_list(metrics_config: Dict[str, Any]) -> List:
            Return the default metrics list for the BB2023 dataset based on the metrics
            configuration.
    """

    def __init__(
        self,
        dataset_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=dataset_config)

        self.eval_config = eval_config or CustomBB2023MOT.get_default_eval_config()
        self.evaluator = None

        self.metrics_config = (
            metrics_config or CustomBB2023MOT.get_default_metrics_config()
        )

    def evaluate(
        self,
        save_dir: Optional[Union[str, Path]] = "eval_result",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the tracker performance on the dataset and return the results.

        Args:
            tracker_names (Optional[List[str]]):
                A list of tracker names to evaluate. If None, all trackers under the
                trackers folder will be evaluated.
            save_dir (Optional[Union[str, Path]]):
                The directory to save the evaluation results. If None, the results will
                not be saved. Defaults to 'eval_result'.

        Returns:
            Dict[str, Dict[str, Any]]:
                The evaluation results in a dictionary format, with tracker names and
                 metric name as keys

        """
        if self.evaluator is None:
            self.evaluator = trackeval.Evaluator(self.eval_config)

        metrics_list = CustomBB2023MOT.get_default_metrics_list(self.metrics_config)
        dataset_list = [self]

        if metrics_list is None or len(metrics_list) == 0:
            raise Exception("No metrics selected for evaluation, check metrics_config.")

        res, _ = self.evaluator.evaluate(
            dataset_list, metrics_list, show_progressbar=True
        )

        # hard coding for class name and dataset type
        cls = "pedestrian"
        dataset_type = dataset_list[0].get_name()
        # reformat results: res[tracker_name][seq_name][metric_name]
        new_res: Dict[str, Dict[str, Any]] = {}
        metric_names = trackeval.utils.validate_metrics_list(metrics_list)
        for tracker_name, _ in res[dataset_type].items():
            new_res[tracker_name] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                table_res = {}
                for seq_name, eval in res[dataset_type][tracker_name].items():
                    table_res[seq_name] = eval[cls][metric_name]
                new_res[tracker_name][metric_name] = metric.summary_results(table_res)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            save_file = save_dir / f"{cur_dt_str()}.json"
            with open(save_file, "w") as f:
                json.dump(new_res, f, indent=2)

        return new_res

    def set_eval_config(self, eval_config: Dict[str, Any]):
        """Set the evaluation configuration with specified values"""
        if self.eval_config is None:
            self.eval_config = CustomBB2023MOT.get_default_eval_config()
        self.eval_config.update(eval_config)
        self.evaluator = None

    def set_metrics_config(self, metrics_config: Dict[str, Any]):
        """Set the metrics configuration with specified values"""
        if self.metrics_config is None:
            self.metrics_config = CustomBB2023MOT.get_default_metrics_config()
        self.metrics_config.update(metrics_config)

    def set_dataset_config(self, dataset_config: Dict[str, Any]):
        """Set the dataset configuration with specified values"""
        if self.config is None:
            self.config = CustomBB2023MOT.get_default_dataset_config()
        self.config.update(dataset_config)

    @staticmethod
    def get_default_dataset_config():
        """Default class config values for testing"""
        dataset_config = (
            trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        )
        dataset_name = "BB2023"
        # get current folder
        base_path = Path(__file__).parent
        dataset_config.update(
            {
                # Filenames of trackers to eval (if None, all in folder)
                "TRACKERS_TO_EVAL": ["gt", "yolov8_bt"],
                "SPLIT_TO_EVAL": "test",  # Valid: 'train', 'test', 'all'
                # location of GT data
                "GT_FOLDER": base_path / f"test_data/gt/{dataset_name}",
                # '{gt_folder}/{seq}/gt/gt.txt'
                "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
                # Trackers location
                "TRACKERS_FOLDER": base_path / f"test_data/trackers/{dataset_name}",
                # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
                "TRACKER_SUB_FOLDER": "data",
                # Names of trackers to display, if None: TRACKERS_TO_EVAL
                "TRACKER_DISPLAY_NAMES": None,
                # Where to save eval results (if None, same as TRACKERS_FOLDER)
                "OUTPUT_FOLDER": None,
                # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
                "OUTPUT_SUB_FOLDER": "",
                # Valid: ['pedestrian'], limitation of using MOT config
                "CLASSES_TO_EVAL": ["pedestrian"],
                # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
                "BENCHMARK": dataset_name,
                # Whether tracker input files are zipped
                "INPUT_AS_ZIP": False,
                # Whether to print current config
                "PRINT_CONFIG": False,
                # Whether to perform preprocessing (never done for MOT15)
                "DO_PREPROC": False,
                # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
                "SEQMAP_FOLDER": None,
                # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
                "SEQMAP_FILE": None,
                # If not None, directly specify sequences to eval and their number of timesteps
                "SEQ_INFO": None,
                # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                # If True, then the middle 'benchmark-split' folder is skipped for both.
                "SKIP_SPLIT_FOL": False,
            }
        )

        return dataset_config

    @staticmethod
    def get_default_eval_config() -> Dict[str, Any]:
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update(
            {
                "TIME_PROGRESS": True,
                # turn on parallel processing
                "USE_PARALLEL": True,
                "NUM_PARALLEL_CORES": 4,
                # reduce print
                "PRINT_RESULTS": True,
                "PRINT_ONLY_COMBINED": True,
                "PRINT_CONFIG": False,
                "DISPLAY_LESS_PROGRESS": True,
                # output
                "OUTPUT_SUMMARY": True,
            }
        )
        return eval_config

    @staticmethod
    def get_default_metrics_config() -> Dict[str, Any]:
        """Default metrics config values"""
        metrics_config = {
            "METRICS": ["HOTA", "CLEAR", "Identity"],
            "THRESHOLD": 0.01,
            "PRINT_CONFIG": False,
        }
        return metrics_config

    @staticmethod
    def get_default_metrics_list(metrics_config: Dict[str, Any]) -> List:
        """Default metrics list"""
        metrics_list = []
        for metric in [
            trackeval.metrics.HOTA,
            trackeval.metrics.CLEAR,
            trackeval.metrics.Identity,
            trackeval.metrics.VACE,
        ]:
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric(metrics_config))

        if len(metrics_list) == 0:
            raise Exception("No metrics selected for evaluation")

        return metrics_list
