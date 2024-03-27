"""Temporary solution to use TrackEval for HOTA, lots of hard coding expected"""

import json
from multiprocessing import freeze_support

from bbtrack_yolo.custom_bb2023_mot import CustomBB2023MOT
from bbtrack_yolo.util import compare_dicts


def test_trackeval():
    """Test the trackeval package"""

    # Arrange
    freeze_support()

    dataset_config = CustomBB2023MOT.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": "tests/data/trackeval/gt/BB2023",
            "TRACKERS_FOLDER": "tests/data/trackeval/trackers/BB2023",
            "TRACKERS_TO_EVAL": [  # folder names under TRACKERS_FOLDER
                "gt",
                "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003",
            ],  # None to eval all trackers
        }
    )

    data_eval = CustomBB2023MOT(dataset_config)

    data_eval.set_eval_config(
        {
            # reduce print
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "DISPLAY_LESS_PROGRESS": True,
            # output
            "OUTPUT_SUMMARY": True,
        }
    )
    answer = json.load(open("tests/data/trackeval/test_answer.json"))

    # Act
    res = data_eval.evaluate()

    # Assert
    assert compare_dicts(res, answer), "The two dictionaries are not equal"


if __name__ == "__main__":
    test_trackeval()
