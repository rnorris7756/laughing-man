"""Shared constants (no heavy imports)."""

from __future__ import annotations

# Haar cascades are poor for pose; BlazeFace is a better default. See:
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

BLAZE_FACE_SHORT_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
BLAZE_FACE_FULL_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)

MODEL_ENV = "LAUGHING_MAN_FACE_MODEL"

# OpenCV YuNet (face_detection_yunet_2023mar.onnx); used when --face-backend yunet.
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_ENV = "LAUGHING_MAN_YUNET_MODEL"

STABLE_IMAGE_NAME = "limg.png"
ROT_IMAGE_NAME = "ltext.png"

MIN_FACE_SIZE = (100, 100)
MIN_DETECTION_CONFIDENCE = 0.45
MIN_SUPPRESSION_THRESHOLD = 0.3

ROT_RESO = 5
DEFAULT_ROI_LAMBDA = 0.55
DEFAULT_SIZE_LAMBDA = 0.55
FADEOUT_LIM = 50
PRIVACY_BLUR_MAX_SIGMA = 28.0
# Linear size ratio for privacy blur (width/height). Blur runs on a downscaled
# frame with sigma scaled by the same factor, then upscaled — much faster than
# full-resolution blur at PRIVACY_BLUR_MAX_SIGMA. Use 1.0 to disable downscaling.
PRIVACY_BLUR_DOWNSCALE = 0.25
DEFAULT_NO_FACE_BLUR_FRAMES = 3
ROI_SCALER = 1.3 * 1.08

# Kalman / optical flow ROI (--roi-motion kalman / kalman_flow)
DEFAULT_ROI_MOTION = "ema"
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 4e-1
FLOW_CENTER_MIX = 0.35
OVERLAY_VERTICAL_SHIFT = 0.10

KEY_WAIT_DELAY_MS = 25
MAIN_WIN_NAME = "Laughing Man (OpenCV)"
TMP_OFFSET = 0.1
LAMBDA_TUNE_STEP = 0.05
_ARROW_KEYS_ROI_UP = frozenset({65362, 2490368})
_ARROW_KEYS_ROI_DOWN = frozenset({65364, 2621440})
_ARROW_KEYS_SIZE_LEFT = frozenset({65361, 2424832})
_ARROW_KEYS_SIZE_RIGHT = frozenset({65363, 2555904})
_QUIT_PREVIEW_KEY_CODES = frozenset(
    {
        3,
        17,
    }
)

CAMERA_INDEX = 0
