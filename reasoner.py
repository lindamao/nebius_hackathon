from __future__ import annotations

import json
import logging
import time
import threading
from openai import OpenAI

import config

SYSTEM_PROMPT = """\
You are an empathetic robot companion observing a human through a camera and microphone.
You receive raw perception data: body pose landmarks (33 keypoints with x,y,z coordinates), \
facial expression scores, and transcribed speech.

Based on all this data, determine what you see, hear, think, and which action to take.

Available actions (pick exactly one):
- cheerful_dance : Perform a cheerful dance to uplift the human's mood if they are sad or worried
- hug            : Open arms and give the human a warm hug if they open their arms to signal hug
- blow_kisses    : Blow kisses to show affection and love if they signal affection
- kungfu_fighting: Practice kung fu fighting with the human if they express the desire to exercise or fight.
- nothing        : Do nothing if the response does not fall strongly into one of the above categories.

IMPORTANT: Keep each field to ONE short sentence (under 20 words). \
Respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "sees": "<brief observation>",
  "hears": "<what the human said, or 'nothing'>",
  "thinks": "<brief interpretation>",
  "action": "<one of: cheerful_dance, hug, blow_kisses, kungfu_fighting, nothing>"
}
"""

MAX_RETRIES = 2


class Reasoner:
    """Calls Nebius LLM with perception data, respecting throttle constraints."""

    def __init__(self):
        self._client = OpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=config.NEBIUS_API_KEY,
        )
        self._last_call_time = 0.0
        self._last_pose = None
        self._last_expression = None
        self._last_speech = ""
        self._lock = threading.Lock()
        self._busy = False

        self.latest_result: dict | None = None

    # --- public API -------------------------------------------------------

    def should_call(self, pose: dict, expression: dict, speech: str) -> bool:
        """Check whether the throttle gate allows a new LLM call."""
        if self._busy:
            return False

        elapsed = time.monotonic() - self._last_call_time
        if elapsed < config.LLM_MIN_INTERVAL_S:
            return False

        new_speech = speech and speech != self._last_speech
        if new_speech:
            return True

        if self._last_expression is not None:
            for key in expression.get("scores", {}):
                old = self._last_expression.get("scores", {}).get(key, 0)
                new = expression["scores"][key]
                if abs(new - old) >= config.EXPRESSION_CHANGE_THRESHOLD:
                    return True

        if self._has_pose_changed(pose):
            return True

        if elapsed >= config.LLM_IDLE_TIMEOUT_S:
            return True

        return False

    def _has_pose_changed(self, pose: dict) -> bool:
        """Detect significant change between current and last raw pose landmarks."""
        if self._last_pose is None or not pose or not self._last_pose:
            return pose != self._last_pose

        total_delta = 0.0
        count = 0
        for name, coords in pose.items():
            prev = self._last_pose.get(name)
            if prev is None:
                continue
            dx = coords["x"] - prev["x"]
            dy = coords["y"] - prev["y"]
            total_delta += (dx * dx + dy * dy) ** 0.5
            count += 1

        if count == 0:
            return False
        avg_delta = total_delta / count
        return avg_delta >= config.POSE_CHANGE_THRESHOLD

    def call(self, pose: dict, expression: dict, speech: str):
        """
        Make a non-blocking LLM call in a background thread.
        Blocks further calls until this one completes.
        """
        self._busy = True
        self._last_call_time = time.monotonic()
        self._last_pose = dict(pose) if pose else {}
        self._last_expression = dict(expression)
        self._last_speech = speech

        threading.Thread(
            target=self._do_call,
            args=(dict(pose) if pose else {}, dict(expression), speech),
            daemon=True,
        ).start()

    def _do_call(self, pose: dict, expression: dict, speech: str):
        perception = {
            "pose_landmarks": pose,
            "expression": expression,
            "speech": speech or "(silence)",
        }
        user_msg = json.dumps(perception, indent=2)

        logging.info("\n" + "-" * 60)
        logging.info("[LLM PROMPT]")
        logging.info("  System: %s...", SYSTEM_PROMPT[:120])
        logging.info("  User:\n%s", user_msg)
        logging.info("-" * 60)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=config.NEBIUS_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_msg,
                                }
                            ],
                        },
                    ],
                    temperature=0.4,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()

                logging.info("\n" + "-" * 60)
                logging.info("[LLM RAW RESPONSE]")
                logging.info(raw)
                logging.info("-" * 60)

                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                result = json.loads(raw)

                if result.get("action") not in config.ACTIONS:
                    result["action"] = config.ACTIONS[0]

                with self._lock:
                    self.latest_result = result

                self._log(result)
                break

            except json.JSONDecodeError as e:
                logging.error("[LLM JSON ERROR] attempt %d/%d: %s", attempt, MAX_RETRIES, e)
                if attempt >= MAX_RETRIES:
                    logging.error("[LLM] Giving up after retries — response was not valid JSON.")
            except Exception as e:
                logging.error("[LLM ERROR] %s", e)
                break

        self._last_call_time = time.monotonic()
        self._busy = False

    def get_latest(self) -> dict | None:
        with self._lock:
            return self.latest_result

    @staticmethod
    def _log(result: dict):
        logging.info("\n" + "=" * 60)
        logging.info("  SEES:   %s", result.get("sees", ""))
        logging.info("  HEARS:  %s", result.get("hears", ""))
        logging.info("  THINKS: %s", result.get("thinks", ""))
        logging.info("  ACTION: %s", result.get("action", ""))
        logging.info("=" * 60 + "\n")
