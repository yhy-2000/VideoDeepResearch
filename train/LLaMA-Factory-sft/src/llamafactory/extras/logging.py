
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

from .constants import RUNNING_LOG


_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


class LoggerHandler(logging.Handler):
    r

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self._formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(filename)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.setLevel(logging.INFO)
        os.makedirs(output_dir, exist_ok=True)
        self.running_log = os.path.join(output_dir, RUNNING_LOG)
        if os.path.exists(self.running_log):
            os.remove(self.running_log)

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _write_log(self, log_entry: str) -> None:
        with open(self.running_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def emit(self, record) -> None:
        if record.name == "httpx":
            return

        log_entry = self._formatter.format(record)
        self.thread_pool.submit(self._write_log, log_entry)

    def close(self) -> None:
        self.thread_pool.shutdown(wait=True)
        return super().close()


class _Logger(logging.Logger):
    r

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_rank0_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)


def _get_default_logging_level() -> "logging._Level":
    r
    env_level_str = os.getenv("LLAMAFACTORY_VERBOSITY", None)
    if env_level_str:
        if env_level_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_level_str.upper()]
        else:
            raise ValueError(f"Unknown logging level: {env_level_str}.")

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "_Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r
    global _default_handler

    with _thread_lock:
        if _default_handler:
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> "_Logger":
    r
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def add_handler(handler: "logging.Handler") -> None:
    r
    _configure_library_root_logger()
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    r
    _configure_library_root_logger()
    _get_library_root_logger().removeHandler(handler)


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once
