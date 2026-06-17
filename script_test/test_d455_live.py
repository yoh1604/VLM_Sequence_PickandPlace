import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align = rs.align(rs.stream.color)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Frame tidak lengkap")
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("D455 RGB", color)
        cv2.imshow("D455 Depth", depth_colormap)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
