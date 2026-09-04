# -*- coding: utf-8 -*-
"""HTTP通讯模块：发送最终跌倒结果，并接收外部重置命令。

通讯方向：

1. 跌倒检测程序向另一个程序发送：
   POST http://127.0.0.1:8080/fall
   JSON：{"fall": true} 或 {"fall": false}

2. 另一个程序向跌倒检测程序发送：
   POST http://127.0.0.1:8123/reset
   JSON：{} 表示重置全部人员；
   JSON：{"person_id": 2} 表示只重置ID为2的人员。

线程安全原则：

- HTTP发送在后台线程执行，不阻塞相机和模型推理。
- HTTP接收线程只把重置命令放入队列。
- main.py主线程读取队列后再清除检测状态。
- HTTP线程不能直接修改跌倒检测器、AI融合器等状态。
"""

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib import request as url_request


class HttpBridge:
    """负责最终结果发送和外部重置命令接收。"""

    def __init__(self, config: dict):
        cfg = config.get("http_bridge") or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.target_url = str(
            cfg.get("target_url", "http://127.0.0.1:8080/fall")
        )
        self.listen_host = str(
            cfg.get("listen_host", "127.0.0.1")
        )
        self.listen_port = int(
            cfg.get("listen_port", 8123)
        )
        self.timeout_s = float(
            cfg.get("timeout_s", 2.0)
        )
        self.retry_count = int(
            cfg.get("retry_count", 2)
        )
        self.retry_interval_s = float(
            cfg.get("retry_interval_s", 0.3)
        )

        if self.timeout_s <= 0.0:
            raise ValueError("http_bridge.timeout_s必须大于0")

        if self.retry_count < 0:
            raise ValueError("http_bridge.retry_count不能小于0")

        if self.retry_interval_s < 0.0:
            raise ValueError(
                "http_bridge.retry_interval_s不能小于0"
            )

        if not 1 <= self.listen_port <= 65535:
            raise ValueError(
                "http_bridge.listen_port必须在1到65535之间"
            )

        # AlertManager产生的状态变化先进入发送队列。
        self._send_queue = queue.Queue(maxsize=20)

        # HTTP服务线程收到的重置命令先进入重置队列。
        # None表示重置全部人员，整数表示重置指定person_id。
        self._reset_queue = queue.Queue()

        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.enabled:
            self._send_thread = threading.Thread(
                target=self._send_worker,
                name="fall-result-sender",
                daemon=True,
            )
            self._send_thread.start()

    # ------------------------------------------------------------------
    # 发送最终结果
    # ------------------------------------------------------------------

    def handle_alert(self, event) -> None:
        """接收AlertManager事件，只提交最终true或false。
        AlertManager只在最终状态变化时调用本函数：

        - FALL_CONFIRMED：发送{"fall": true}
        - FALL_RECOVERED：发送{"fall": false}
        - 状态没有变化：AlertManager不会调用本函数

        本函数只把任务放进队列，不进行网络等待。
        """

        if not self.enabled:
            return

        is_fall = event.event_type == "FALL_CONFIRMED"

        try:
            self._send_queue.put_nowait(is_fall)
        except queue.Full:
            print(
                "[http_bridge] 发送队列已满，"
                f"本次结果未发送：fall={is_fall}"
            )

    def _send_worker(self) -> None:
        """后台线程持续读取发送队列并执行HTTP请求。"""

        while not self._stop_event.is_set():
            try:
                is_fall = self._send_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._post_fall_result(bool(is_fall))
            finally:
                self._send_queue.task_done()

    def _post_fall_result(self, is_fall: bool) -> bool:
        """发送一次最终结果，失败时按配置进行重试。"""

        payload = {
            "fall": bool(is_fall),
        }
        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        total_attempts = self.retry_count + 1

        for attempt in range(total_attempts):
            try:
                http_request = url_request.Request(
                    self.target_url,
                    data=data,
                    method="POST",
                    headers={
                        "Content-Type":
                            "application/json; charset=utf-8"
                    },
                )

                with url_request.urlopen(
                    http_request,
                    timeout=self.timeout_s,
                ) as response:
                    # 读取响应，让连接能够正常释放。
                    # 当前不通过本次响应重置，而是使用独立/reset接口。
                    response.read()

                    if not 200 <= response.status < 300:
                        raise RuntimeError(
                            f"HTTP状态码异常：{response.status}"
                        )

                print(
                    "[http_bridge] 最终结果发送成功："
                    f"fall={is_fall}"
                )
                return True

            except Exception as error:
                current_attempt = attempt + 1

                if current_attempt >= total_attempts:
                    print(
                        "[http_bridge] 最终结果发送失败，"
                        "本地跌倒检测继续运行："
                        f"{type(error).__name__}: {error}"
                    )
                    return False

                print(
                    "[http_bridge] 发送失败，准备重试："
                    f"{current_attempt}/{total_attempts}，"
                    f"{type(error).__name__}: {error}"
                )
                time.sleep(self.retry_interval_s)

        return False

    # ------------------------------------------------------------------
    # 接收外部重置命令
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """启动/reset接口，接收另一个程序发来的重置命令。"""

        if not self.enabled or self._server is not None:
            return

        bridge = self

        class Handler(BaseHTTPRequestHandler):
            """处理另一个程序发送的HTTP请求。"""

            def _send_json(
                self,
                status_code: int,
                response_data: dict,
            ) -> None:
                """向调用方返回JSON响应。"""

                body = json.dumps(
                    response_data,
                    ensure_ascii=False,
                ).encode("utf-8")

                self.send_response(status_code)
                self.send_header(
                    "Content-Type",
                    "application/json; charset=utf-8",
                )
                self.send_header(
                    "Content-Length",
                    str(len(body)),
                )
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                """只允许POST请求执行重置操作。"""

                request_path = self.path.split("?", 1)[0].rstrip("/")

                if request_path != "/reset":
                    self._send_json(
                        404,
                        {
                            "ok": False,
                            "error": "unknown path",
                        },
                    )
                    return

                try:
                    content_length = int(
                        self.headers.get("Content-Length", "0")
                    )

                    if content_length < 0:
                        raise ValueError(
                            "Content-Length不能小于0"
                        )

                    request_data = {}

                    if content_length > 0:
                        raw_body = self.rfile.read(content_length)
                        request_data = json.loads(
                            raw_body.decode("utf-8")
                        )

                        if not isinstance(request_data, dict):
                            raise ValueError(
                                "请求正文必须是JSON对象"
                            )

                    # 没有person_id表示重置全部人员。
                    person_id = request_data.get("person_id")

                    if person_id is not None:
                        person_id = int(person_id)

                    # 只将命令放入线程安全队列。
                    # 真正的状态清理由main.py主线程执行。
                    bridge._reset_queue.put(person_id)

                    target_text = (
                        "全部人员"
                        if person_id is None
                        else f"person_id={person_id}"
                    )
                    print(
                        "[http_bridge] 已收到重置请求："
                        f"{target_text}"
                    )

                    self._send_json(
                        202,
                        {
                            "ok": True,
                            "accepted": True,
                            "person_id": person_id,
                        },
                    )

                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as error:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": str(error),
                        },
                    )

                except Exception as error:
                    self._send_json(
                        500,
                        {
                            "ok": False,
                            "error":
                                f"{type(error).__name__}: {error}",
                        },
                    )

            def do_GET(self) -> None:
                """GET不能执行重置，防止浏览器误访问导致状态清空。"""

                self._send_json(
                    405,
                    {
                        "ok": False,
                        "error": "reset only accepts POST",
                    },
                )
            def log_message(self, *args) -> None:
                """关闭BaseHTTPRequestHandler默认访问日志。"""

                return

        try:
            self._server = ThreadingHTTPServer(
                (self.listen_host, self.listen_port),
                Handler,
            )
        except OSError as error:
            self._server = None
            raise RuntimeError(
                "HTTP重置接口启动失败："
                f"{self.listen_host}:{self.listen_port}，{error}"
            ) from error

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="fall-reset-server",
            daemon=True,
        )
        self._server_thread.start()

        print(
            "[http_bridge] 重置接口已启动："
            f"http://{self.listen_host}:"
            f"{self.listen_port}/reset"
        )

    def poll_reset_requests(self) -> list:
        """返回所有待处理重置命令。
        返回列表中的含义：
        - None：重置全部人员；
        - int：只重置对应person_id。

        本方法应当只由main.py相机主线程调用。
        """
        reset_requests = []

        while True:
            try:
                reset_requests.append(
                    self._reset_queue.get_nowait()
                )
            except queue.Empty:
                break

        return reset_requests

    # ------------------------------------------------------------------
    # 关闭通讯
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """关闭HTTP服务和后台发送线程。"""
        self._stop_event.set()

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        if self._send_thread is not None:
            self._send_thread.join(timeout=2.0)
            self._send_thread = None