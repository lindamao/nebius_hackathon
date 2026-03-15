#!/usr/bin/env python3
"""
Real-time perception-to-action demo.
Camera + Mic → Pose / Expression / Speech → Nebius LLM → Robot Action
"""

import cv2
import numpy as np
import time
import signal
import sys
import os
import logging
from datetime import datetime

import config
from vision import PoseExtractor, FaceExpressionExtractor
from audio import AudioTranscriber
from reasoner import Reasoner


def draw_overlay(frame, expression, speech, llm_result):
    """Draw perception + LLM output on the OpenCV frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent bar at top
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    # Semi-transparent bar at bottom
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top-left: expression
    expr_text = f"Expression: {expression.get('dominant', '?')}"
    scores = expression.get("scores", {})
    score_parts = [f"{k}={v:.1f}" for k, v in scores.items() if v > 0.1]
    if score_parts:
        expr_text += f"  ({', '.join(score_parts)})"
    cv2.putText(frame, expr_text, (15, 35), font, 0.7, (0, 255, 200), 2)

    # Top-right: current action
    if llm_result:
        action = llm_result.get("action", "")
        action_color = {
            "cheerful_dance": (0, 200, 255),
            "hug": (255, 150, 50),
            "blow_kisses": (180, 50, 255),
            "kungfu_fighting": (0, 150, 255),
        }.get(action, (255, 255, 255))
        action_label = f"ACTION: {action}"
        text_size = cv2.getTextSize(action_label, font, 0.9, 2)[0]
        cv2.putText(frame, action_label, (w - text_size[0] - 15, 35),
                    font, 0.9, action_color, 2)

        thinks = llm_result.get("thinks", "")
        if thinks:
            max_chars = w // 12
            thinks_trunc = thinks[:max_chars] + ("..." if len(thinks) > max_chars else "")
            cv2.putText(frame, f"Thinks: {thinks_trunc}", (15, 70),
                        font, 0.55, (200, 200, 200), 1)

    # Bottom: speech
    speech_display = speech if speech else "(listening...)"
    max_chars = w // 11
    speech_trunc = speech_display[-max_chars:]
    cv2.putText(frame, f"Speech: {speech_trunc}", (15, h - 60),
                font, 0.65, (255, 255, 255), 2)

    # Bottom-right: status indicator
    status = "LIVE"
    cv2.circle(frame, (w - 30, h - 70), 8, (0, 0, 255), -1)
    cv2.putText(frame, status, (w - 80, h - 63), font, 0.5, (0, 0, 255), 1)

    return frame


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.info(f"Logging to {log_file}")
    return log_file


def main():
    setup_logging()
    logging.info("[INIT] Starting perception pipeline...")

    # --- Init modules ---
    pose_extractor = PoseExtractor()
    face_extractor = FaceExpressionExtractor()
    audio_transcriber = AudioTranscriber()
    reasoner = Reasoner()

    if not config.NEBIUS_API_KEY:
        logging.warning("NEBIUS_API_KEY not set — LLM calls will fail.")
        logging.warning("Copy .env.example to .env and add your key.")

    # --- Start audio ---
    logging.info("[INIT] Loading Whisper model (this may take a moment)...")
    audio_transcriber.start()
    logging.info("[INIT] Audio transcriber started.")

    # --- Open camera ---
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        logging.error("Cannot open camera. Check permissions in System Settings > Privacy > Camera.")
        sys.exit(1)

    logging.info("[INIT] Camera opened. Press 'q' to quit.")
    logging.info("-" * 60)

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # --- Main loop ---
    current_pose = {}
    current_expression = {"dominant": "neutral", "scores": {}}

    while running:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read camera frame.")
            time.sleep(0.1)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose extraction
        pose_results, pose_summary = pose_extractor.process(frame_rgb)
        current_pose = pose_summary
        pose_extractor.draw(frame, pose_results)

        # Face expression extraction
        expression_summary = face_extractor.process(frame_rgb)
        current_expression = expression_summary

        # Speech
        speech = audio_transcriber.latest_text

        # LLM reasoning (throttled)
        if reasoner.should_call(current_pose, current_expression, speech):
            reasoner.call(current_pose, current_expression, speech)
            audio_transcriber.clear_text()

        llm_result = reasoner.get_latest()

        # Draw overlay
        frame = draw_overlay(frame, current_expression, speech, llm_result)

        cv2.imshow("Perception Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    logging.info("[SHUTDOWN] Cleaning up...")
    audio_transcriber.stop()
    pose_extractor.close()
    face_extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    logging.info("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main()
