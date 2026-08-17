# pyre-strict
import os
import shutil
from typing import Any, Dict

import sys

import opensfm.dataset
import yaml
from opensfm import io


try:
    from libfb.py import parutil

    DATA_PATH: str = parutil.get_dir_path("mapillary/opensfm/opensfm/test/data")
except ImportError:
    DATA_PATH = os.path.abspath("data")


def create_berlin_test_folder(tmpdir: Any) -> opensfm.dataset.DataSet:
    src = os.path.join(DATA_PATH, "berlin")
    dst = str(tmpdir.mkdir("berlin"))
    files = ["images", "masks", "config.yaml", "ground_control_points.json"]
    for filename in files:
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)
        if sys.platform == "win32":
            if os.path.isdir(src_path):
                os.mkdir(dst_path)
                for f in os.listdir(src_path):
                    shutil.copyfile(os.path.join(src_path, f), os.path.join(dst_path, f))
            else:
                shutil.copyfile(src_path, dst_path)
        else:
            os.symlink(src_path, dst_path)
    return opensfm.dataset.DataSet(dst)


def save_config(config: Dict[str, Any], path: str) -> None:
    with io.open_wt(os.path.join(path, "config.yaml")) as fout:
        yaml.safe_dump(config, fout, default_flow_style=False)
