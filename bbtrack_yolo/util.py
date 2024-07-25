import os
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger


def cur_dt_str():
    """get current datetime in string format
    :return: current datetime in string format
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def compare_dicts(d1, d2):
    """Compare two dictionaries"""

    def almost_equal(a, b, rel_tol=1e-9, abs_tol=0.0):
        """Check if two floats are almost equal"""
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    if d1.keys() != d2.keys():
        return False
    for k in d1:
        if isinstance(d1[k], float) and isinstance(d2[k], float):
            if not almost_equal(d1[k], d2[k]):
                return False
        elif d1[k] != d2[k]:
            return False
    return True


def get_default_tqdm_args(
    additional_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get default tqdm arguments"""
    default = {
        "bar_format": "{l_bar}{bar:10}{r_bar}",
    }
    default.update(additional_args or {})
    return default


def init_logger(
    level: str = "INFO",
    to_file: bool = False,
    verbose: bool = True,
    log_file_notion: str = None,
):
    """initialize logger
    :param level: log level
    :param to_file: whether to save log to file
    :param verbose: whether to show logger initialization info
    :param log_file_notion: notion to add to log file name
    """

    level = level.upper()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n"
        "<level>{message}</level>"
    )
    logger.remove()
    tqdm_enabled = False
    # if tqdm is in the environment, use it to write logs
    try:
        from tqdm.auto import tqdm

        logger.add(
            lambda msg: tqdm.write(msg, end=""), colorize=True, format=fmt, level=level
        )
        tqdm_enabled = True
    except ImportError:
        logger.add(
            lambda msg: print(msg, end=""), colorize=True, format=fmt, level=level
        )
    logger_file_path: str = ""
    if to_file:
        os.makedirs("logs", exist_ok=True)
        notion = f"_{log_file_notion}" if log_file_notion else ""
        logger_file_path = f"logs/{cur_dt_str()}_{notion}.log"
        logger.add(logger_file_path, level="DEBUG", rotation="500 MB")
    if verbose:
        log_str = f"logger initialized with {level} level"
        if to_file:
            log_str += f", log file at {logger_file_path} with DEBUG level"
        if tqdm_enabled:
            log_str += ", tqdm.write() enabled"
        logger.info(log_str)


def args_in_lines(args: Namespace) -> str:
    """convert argparse args to string
    :param args: argparse args
    :return: string of args
    """
    return "\n".join(f"{arg}: {getattr(args, arg)}" for arg in vars(args))
