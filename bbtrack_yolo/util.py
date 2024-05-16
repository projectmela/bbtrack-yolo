from datetime import datetime
from typing import Any, Dict, Optional


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
