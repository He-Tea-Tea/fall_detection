# -*- coding: utf-8 -*-
"""接收跌倒检测结果，并可向跌倒检测程序发送重置命令。

接收地址：
    POST http://127.0.0.1:8080/fall

接收内容：
    {"fall": true}
    {"fall": false}

发送重置：
    POST http://127.0.0.1:8123/reset

运行：
    python receiver.py

终端命令：
    reset       重置全部人员
    reset 2     重置person_id=2
    status      查看最后一次跌倒结果
    quit        退出程序
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib import request as url_request


# 本程序接收跌倒结果的地址。
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8080

# 跌倒检测程序接收重置命令的地址。
RESET_URL = "http://127.0.0.1:8123/reset"

# 保存最近一次收到的最终结果。
latest_fall_result: Optional[bool] = None
result_lock = threading.Lock()


def handle_fall_result(is_fall: bool) -> None:
    """收到最终结果后执行后续业务。

    后续接入另一个AI模型、语音对话、机器人控制或告警模块时，
    把相应调用写在这里即可。
    """

    if is_fall:
        print("【警报】收到最终结果：有人跌倒")

        # 后续接口示例：
        # voice_module.ask_person_status()
        # robot_controller.stop()
        # alarm_service.send_alarm()
        # another_ai.handle_fall(True)

    else:
        print("【恢复】收到最终结果：当前没有人跌倒")

        # 后续接口示例：
        # alarm_service.cancel_alarm()
        # another_ai.handle_fall(False)


def send_reset(person_id: Optional[int] = None) -> bool:
    """向跌倒检测程序发送重置命令。

    person_id=None：重置全部人员。
    person_id=2：只重置ID为2的人员。
    """

    payload = {}

    if person_id is not None:
        payload["person_id"] = int(person_id)

    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    http_request = url_request.Request(
        RESET_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )

    try:
        with url_request.urlopen(
            http_request,
            timeout=2.0,
        ) as response:
            response_text = response.read().decode("utf-8")
            response_data = (
                json.loads(response_text)
                if response_text
                else {}
            )

        if response_data.get("accepted") is True:
            if person_id is None:
                print("重置命令发送成功：重置全部人员")
            else:
                print(
                    "重置命令发送成功："
                    f"person_id={person_id}"
                )
            return True

        print(
            "跌倒检测程序没有接受重置命令："
            f"{response_data}"
        )
        return False

    except Exception as error:
        print(
            "发送重置命令失败："
            f"{type(error).__name__}: {error}"
        )
        return False


class FallResultHandler(BaseHTTPRequestHandler):
    """处理跌倒检测程序发来的HTTP请求。"""

    def _send_json(
        self,
        status_code: int,
        response_data: dict,
    ) -> None:
        """返回JSON响应。"""

        response_body = json.dumps(
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
            str(len(response_body)),
        )
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self) -> None:
        """接收POST /fall请求。"""

        request_path = self.path.split("?", 1)[0].rstrip("/")

        if request_path != "/fall":
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

            if content_length <= 0:
                raise ValueError("请求中没有JSON数据")

            raw_body = self.rfile.read(content_length)
            request_data = json.loads(
                raw_body.decode("utf-8")
            )

            if not isinstance(request_data, dict):
                raise ValueError("请求内容必须是JSON对象")

            if "fall" not in request_data:
                raise ValueError("请求中缺少fall字段")

            # 严格检查布尔类型，避免字符串"false"被误认为True。
            is_fall = request_data["fall"]

            if not isinstance(is_fall, bool):
                raise ValueError(
                    "fall必须是JSON布尔值true或false，"
                    "不能使用字符串"
                )

            global latest_fall_result

            # 用锁保护共享变量，避免多个HTTP线程同时修改。
            with result_lock:
                latest_fall_result = is_fall

            # 先确认收到消息，避免后续业务处理拖慢HTTP响应。
            self._send_json(
                200,
                {
                    "ok": True,
                    "received": True,
                },
            )

            # 在独立线程中执行后续业务。
            threading.Thread(
                target=handle_fall_result,
                args=(is_fall,),
                name="fall-result-handler",
                daemon=True,
            ).start()

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
        """GET只用于查看服务状态，不改变判断结果。"""

        request_path = self.path.split("?", 1)[0].rstrip("/")

        if request_path != "/status":
            self._send_json(
                404,
                {
                    "ok": False,
                    "error": "unknown path",
                },
            )
            return

        with result_lock:
            current_result = latest_fall_result

        self._send_json(
            200,
            {
                "ok": True,
                "fall": current_result,
            },
        )

    def log_message(self, *args) -> None:
        """关闭默认HTTP访问日志，避免终端刷屏。"""

        return


def command_loop() -> None:
    """处理终端输入，用于测试主动重置接口。"""

    while True:
        try:
            command = input(
                "\n请输入命令"
                "（status/reset/reset ID/quit）："
            ).strip()

        except (EOFError, KeyboardInterrupt):
            return

        if not command:
            continue

        parts = command.split()
        action = parts[0].lower()

        if action == "status":
            with result_lock:
                current_result = latest_fall_result

            if current_result is None:
                print("目前还没有收到跌倒检测结果")
            else:
                print(
                    "最近结果："
                    f"fall={current_result}"
                )

        elif action == "reset":
            if len(parts) == 1:
                send_reset()
                continue

            try:
                person_id = int(parts[1])
            except ValueError:
                print("person_id必须是整数")
                continue

            send_reset(person_id)

        elif action in {"quit", "exit"}:
            return

        else:
            print(
                "未知命令，可使用："
                "status、reset、reset 2、quit"
            )


def main() -> None:
    """启动结果接收服务和终端命令循环。"""

    try:
        server = ThreadingHTTPServer(
            (LISTEN_HOST, LISTEN_PORT),
            FallResultHandler,
        )
    except OSError as error:
        raise RuntimeError(
            "接收服务启动失败："
            f"{LISTEN_HOST}:{LISTEN_PORT}，"
            "请检查端口是否被占用。"
            f"原始错误：{error}"
        ) from error

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="fall-result-server",
        daemon=True,
    )
    server_thread.start()

    print(
        "跌倒结果接收服务已启动："
        f"http://{LISTEN_HOST}:{LISTEN_PORT}/fall"
    )
    print(
        "跌倒检测重置地址："
        f"{RESET_URL}"
    )

    try:
        command_loop()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        print("接收服务已关闭")


if __name__ == "__main__":
    main()