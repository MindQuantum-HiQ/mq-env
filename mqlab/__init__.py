"""课程实验的小工具：环境检查、练习的即时反馈与数据路径。

在 Notebook 中这样使用::

    from mqlab import check
    check.bell_states(bell)
"""
import os

# MindSpore 默认会打印大量 INFO / WARNING 日志，只保留错误信息
os.environ.setdefault("GLOG_v", "3")

from .env import check_env, data_path  # noqa: E402
from . import check  # noqa: E402

__all__ = ["check", "check_env", "data_path"]
