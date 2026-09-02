from pyorbbecsdk import Pipeline


def main():
    pipeline = Pipeline()

    try:
        print("1. 创建 Pipeline")
        pipeline.start()
        print("2. Pipeline 启动成功")

        print("3. 等待图像帧...")

        frames = None
        for i in range(10):
            frames = pipeline.wait_for_frames(3000)

            print(f"   第 {i + 1} 次获取帧: {frames is not None}")

            if frames is not None:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                print(
                    f"   Color: {color_frame is not None}, "
                    f"Depth: {depth_frame is not None}"
                )

                if color_frame is not None or depth_frame is not None:
                    break

        if frames is None:
            print("错误：一直没有获取到 frames")
            return

        print("\n========== 当前帧 ==========")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is not None:
            print("Color:")
            print("  width :", color_frame.get_width())
            print("  height:", color_frame.get_height())
            print("  format:", color_frame.get_format())

        if depth_frame is not None:
            print("Depth:")
            print("  width :", depth_frame.get_width())
            print("  height:", depth_frame.get_height())
            print("  format:", depth_frame.get_format())

        print("\n========== Camera Parameters ==========")

        camera_param = pipeline.get_camera_param()

        print("camera_param =", camera_param)

        print("\n--- RGB Intrinsic ---")
        rgb = camera_param.rgb_intrinsic
        print("width :", rgb.width)
        print("height:", rgb.height)
        print("fx    :", rgb.fx)
        print("fy    :", rgb.fy)
        print("cx    :", rgb.cx)
        print("cy    :", rgb.cy)

        print("\n--- Depth Intrinsic ---")
        depth = camera_param.depth_intrinsic
        print("width :", depth.width)
        print("height:", depth.height)
        print("fx    :", depth.fx)
        print("fy    :", depth.fy)
        print("cx    :", depth.cx)
        print("cy    :", depth.cy)

        print("\n--- Depth Scale ---")
        try:
            print(depth_frame.get_depth_scale())
        except Exception as e:
            print("读取 depth scale 失败:", e)

        print("\n--- Depth -> RGB Transform ---")
        print(camera_param.transform)

    except Exception as e:
        print("发生异常:")
        print(type(e).__name__, e)

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()