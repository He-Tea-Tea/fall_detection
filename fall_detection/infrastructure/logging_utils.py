# -*- coding: utf-8 -*-
"""跌倒检测系统统一日志工具。

所有模块通过logging.getLogger(__name__)获取日志器，由main.py在启动时调用
configure_logging()统一配置终端输出和轮转文件。相同高频告警可通过
log_rate_limiter限频，避免相机异常时每帧刷屏。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time
from typing import Dict, Optional

_CONFIGURED = False
_CONFIGURE_LOCK = threading.Lock()


class LogRateLimiter:
    """记录每类日志上次输出时间，控制重复日志的最短间隔。"""

    def __init__(self):
        self._last_log_s: Dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(
        self,
        key: str,
        interval_s: float,
        now_s: Optional[float] = None,
    ) -> bool:
        """达到输出间隔返回True并更新时间，否则返回False。"""
        now = time.monotonic() if now_s is None else float(now_s)
        interval = max(0.0, float(interval_s))
        with self._lock:
            previous = self._last_log_s.get(str(key))
            if previous is not None and now - previous < interval:
                return False
            self._last_log_s[str(key)] = now
            return True

    def reset(self, key: Optional[str] = None) -> None:
        """清除一个日志键或全部限频历史，主要用于测试。"""
        with self._lock:
            if key is None:
                self._last_log_s.clear()
            else:
                self._last_log_s.pop(str(key), None)


log_rate_limiter = LogRateLimiter()


def configure_logging(config: dict, base_directory: Path) -> logging.Logger:
    """根据config.yaml配置根日志器；重复调用不会重复添加handler。"""
    global _CONFIGURED
    with _CONFIGURE_LOCK:
        if _CONFIGURED:
            return logging.getLogger("fall_detection")

        cfg = config.get("logging") or {}
        level_name = str(cfg.get("level", "INFO")).upper()
        level = getattr(logging, level_name, None)
        if not isinstance(level, int):
            raise ValueError("logging.level必须是DEBUG、INFO、WARNING、ERROR或CRITICAL")

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if bool(cfg.get("console_enabled", True)):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        if bool(cfg.get("enabled", True)):
            log_directory = Path(cfg.get("directory", "Log"))
            if not log_directory.is_absolute():
                log_directory = Path(base_directory) / log_directory
            log_directory.mkdir(parents=True, exist_ok=True)
            max_size_mb = float(cfg.get("max_file_size_mb", 10.0))
            backup_count = int(cfg.get("backup_count", 5))
            if max_size_mb <= 0.0:
                raise ValueError("logging.max_file_size_mb必须大于0")
            if backup_count < 0:
                raise ValueError("logging.backup_count不能小于0")
            log_file = log_directory / str(cfg.get("filename", "fall_detection.log"))
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=int(max_size_mb * 1024 * 1024),
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _CONFIGURED = True
        logger = logging.getLogger("fall_detection")
        logger.info("日志系统初始化完成：level=%s", level_name)
        if bool(cfg.get("enabled", True)):
            logger.info("应用日志文件：%s", log_file.resolve())
        return logger
