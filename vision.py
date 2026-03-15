import os
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class PoseExtractor:
    """Extracts body pose landmarks and summarises arm/posture state."""

    def __init__(self):
        base_opts = BaseOptions(
            model_asset_path=os.path.join(_MODEL_DIR, "pose_landmarker_lite.task"),
        )
        options = PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, frame_rgb):
        """Return (result, summary_dict)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33  # ~30fps in ms
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        summary = self._summarise(result)
        return result, summary

    def draw(self, frame_bgr, result):
        if not result.pose_landmarks:
            return
        landmarks = result.pose_landmarks[0]
        h, w = frame_bgr.shape[:2]

        connections = PoseLandmarksConnections.POSE_LANDMARKS
        point_spec = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        line_spec = DrawingSpec(color=(0, 200, 0), thickness=2)

        # Draw connections
        for conn in connections:
            start = landmarks[conn.start]
            end = landmarks[conn.end]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame_bgr, (x1, y1), (x2, y2),
                     line_spec.color, line_spec.thickness)

        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy),
                       point_spec.circle_radius, point_spec.color,
                       point_spec.thickness)

    def _lm(self, landmarks, idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    _LANDMARK_NAMES = {v: k for k, v in PoseLandmark.__members__.items()}

    def _summarise(self, result):
        if not result.pose_landmarks:
            return {}

        landmarks = result.pose_landmarks[0]
        raw = {}
        for idx, lm in enumerate(landmarks):
            name = self._LANDMARK_NAMES.get(idx, str(idx))
            raw[name] = {
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
            }
        return raw

    def close(self):
        self._landmarker.close()


class FaceExpressionExtractor:
    """Extracts facial blendshape scores from MediaPipe FaceLandmarker."""

    def __init__(self):
        base_opts = BaseOptions(
            model_asset_path=os.path.join(_MODEL_DIR, "face_landmarker.task"),
        )
        options = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, frame_rgb):
        """Return summary dict with dominant expression and key scores."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        return self._summarise(result)

    def _summarise(self, result):
        default = {"dominant": "neutral", "scores": {}}

        if not result.face_blendshapes:
            if not result.face_landmarks:
                return default
            return self._geometric_fallback(result.face_landmarks[0])

        blendshapes = result.face_blendshapes[0]
        bs_map = {b.category_name: b.score for b in blendshapes}

        smile = (bs_map.get("mouthSmileLeft", 0) + bs_map.get("mouthSmileRight", 0)) / 2
        frown = (bs_map.get("mouthFrownLeft", 0) + bs_map.get("mouthFrownRight", 0)) / 2
        brow_furrow = (
            bs_map.get("browDownLeft", 0) + bs_map.get("browDownRight", 0)
            + bs_map.get("browInnerUp", 0)
        ) / 3
        surprise = bs_map.get("jawOpen", 0)

        scores = {
            "smile": round(float(smile), 2),
            "frown": round(float(frown), 2),
            "brow_furrow": round(float(brow_furrow), 2),
            "surprise": round(float(surprise), 2),
        }

        best_label, best_val = "neutral", 0.0
        for label, val in [("happy", smile), ("sad", frown),
                           ("worried", brow_furrow), ("surprised", surprise)]:
            if val > best_val and val > 0.2:
                best_label, best_val = label, val

        return {"dominant": best_label, "scores": scores}

    def _geometric_fallback(self, face_landmarks):
        """Approximate expressions from landmark geometry when blendshapes unavailable."""
        def pt(idx):
            lm = face_landmarks[idx]
            return np.array([lm.x, lm.y])

        mouth_left = pt(61)
        mouth_right = pt(291)
        mouth_top = pt(13)
        mouth_bottom = pt(14)
        mouth_w = np.linalg.norm(mouth_right - mouth_left)
        mouth_h = np.linalg.norm(mouth_bottom - mouth_top)

        smile_ratio = mouth_w / (mouth_h + 1e-6)
        smile = float(np.clip((smile_ratio - 3.0) / 4.0, 0, 1))
        frown_val = float(np.clip(
            ((mouth_left[1] + mouth_right[1]) / 2 - (mouth_top[1] + mouth_bottom[1]) / 2) * 30,
            0, 1))
        surprise = float(np.clip((mouth_h / (mouth_w + 1e-6) - 0.35) / 0.3, 0, 1))

        scores = {
            "smile": round(smile, 2),
            "frown": round(frown_val, 2),
            "brow_furrow": 0.0,
            "surprise": round(surprise, 2),
        }

        best_label, best_val = "neutral", 0.0
        for label, val in [("happy", smile), ("sad", frown_val), ("surprised", surprise)]:
            if val > best_val and val > 0.25:
                best_label, best_val = label, val

        return {"dominant": best_label, "scores": scores}

    def close(self):
        self._landmarker.close()
