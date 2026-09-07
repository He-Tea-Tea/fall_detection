# export_onnx.py
from ultralytics import YOLO


def export_model(model_path: str) -> None:
    """导出固定640输入、batch=1的FP32 ONNX。"""

    model = YOLO(model_path)

    output = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=True,
        half=False,
    )

    print(f"导出完成：{output}")


if __name__ == "__main__":
    export_model("yolo26s-pose.pt")
    export_model("yolo26s-seg.pt")