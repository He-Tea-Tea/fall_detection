# -*- coding: utf-8 -*-
"""最终跌倒状态音频提示。

只响应AlertManager产生的最终状态变化：
FALL_CONFIRMED播放跌倒音频，FALL_RECOVERED播放恢复音频。
音频在后台线程播放，不阻塞相机采集、YOLO推理和AI请求。
"""

import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioAlertPlayer:
    """跨Windows和Linux播放最终状态提示音。"""

    def __init__(self, config: dict, base_dir: Path):
        audio_cfg = config.get("audio", {})
        self.enabled = bool(audio_cfg.get("enabled", False))
        self.play_recovery = bool(audio_cfg.get("play_recovery", True))
        self.base_dir = Path(base_dir).resolve()
        self.fall_file = self._resolve_path(
            str(audio_cfg.get("fall_file", "audio/fall_true.wav"))
        )
        self.recovery_file = self._resolve_path(
            str(audio_cfg.get("recovery_file", "audio/fall_false.wav"))
        )
        self._play_lock = threading.Lock()

        if not self.enabled:
            logger.info("音频提示已关闭")
            return

        if not self.fall_file.is_file():
            logger.warning("跌倒音频不存在：%s", self.fall_file)
        if self.play_recovery and not self.recovery_file.is_file():
            logger.warning("恢复音频不存在：%s", self.recovery_file)

        if sys.platform.startswith("linux") and shutil.which("aplay") is None:
            logger.warning(
                "Linux没有找到aplay，请安装alsa-utils："
                "sudo apt install alsa-utils"
            )

    def _resolve_path(self, value: str) -> Path:
        """将相对音频路径按项目目录转换为绝对路径。"""
        path = Path(value)
        return path.resolve() if path.is_absolute() else self.base_dir / path

    def handle_event(self, event) -> None:
        """接收AlertManager事件，只在最终状态变化时播放。"""
        if not self.enabled:
            return

        if event.event_type == "FALL_CONFIRMED":
            self.play(self.fall_file, "FALL")
        elif event.event_type == "FALL_RECOVERED" and self.play_recovery:
            self.play(self.recovery_file, "NO_FALL")

    def play(self, audio_path: Path, label: str) -> None:
        """启动后台播放线程，避免音频阻塞视觉主循环。"""
        if not audio_path.is_file():
            logger.warning("无法播放%s音频，文件不存在：%s", label, audio_path)
            return

        thread = threading.Thread(
            target=self._play_worker,
            args=(audio_path, label),
            name=f"audio-{label.lower()}",
            daemon=True,
        )
        thread.start()

    def _play_worker(self, audio_path: Path, label: str) -> None:
        """根据操作系统选择播放器，同一时间只播放一个提示音。"""
        with self._play_lock:
            try:
                if sys.platform.startswith("win"):
                    self._play_windows(audio_path)
                elif sys.platform.startswith("linux"):
                    self._play_linux(audio_path)
                else:
                    logger.warning("当前操作系统暂不支持音频提示：%s", sys.platform)
                    return

                logger.info("最终状态音频已播放：label=%s file=%s", label, audio_path.name)
            except Exception:
                logger.exception("播放最终状态音频失败：%s", audio_path)

    @staticmethod
    def _play_windows(audio_path: Path) -> None:
        """Windows使用Python自带winsound播放WAV文件。"""
        import winsound

        winsound.PlaySound(
            str(audio_path),
            winsound.SND_FILENAME | winsound.SND_NODEFAULT,
        )

    @staticmethod
    def _play_linux(audio_path: Path) -> None:
        """Ubuntu使用aplay播放WAV文件。"""
        if shutil.which("aplay") is None:
            raise RuntimeError("没有找到aplay，请执行sudo apt install alsa-utils")

        result = subprocess.run(
            ["aplay", "-q", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or f"退出码{result.returncode}"
            raise RuntimeError(f"aplay播放失败：{error}")