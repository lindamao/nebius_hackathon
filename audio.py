from __future__ import annotations

import logging
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import config


class AudioTranscriber:
    """
    Background thread that captures mic audio, detects speech via energy VAD,
    and transcribes completed utterances with faster-whisper.
    """

    def __init__(self):
        self._model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        self._sample_rate = config.AUDIO_SAMPLE_RATE
        self._block_dur = config.AUDIO_BLOCK_DURATION_S
        self._block_size = int(self._sample_rate * self._block_dur)
        self._silence_thresh = config.SILENCE_THRESHOLD
        self._silence_timeout = config.SILENCE_TIMEOUT_S
        self._max_buffer_s = config.MAX_SPEECH_BUFFER_S

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._latest_text = ""
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    # --- public API -------------------------------------------------------

    @property
    def latest_text(self) -> str:
        with self._lock:
            return self._latest_text

    def clear_text(self):
        with self._lock:
            self._latest_text = ""

    def start(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            blocksize=self._block_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread is not None:
            self._thread.join(timeout=3)

    # --- internals --------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_q.put(indata[:, 0].copy())

    def _process_loop(self):
        """Accumulate audio while speech is detected, transcribe on silence or max duration."""
        speech_buffer: list[np.ndarray] = []
        silence_start: float | None = None
        buffer_start: float | None = None
        energy_log_counter = 0

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            energy = float(np.sqrt(np.mean(chunk ** 2)))

            energy_log_counter += 1
            if energy_log_counter % 20 == 0:
                logging.debug("[AUDIO] energy=%.4f  threshold=%.4f  buffer_chunks=%d",
                              energy, self._silence_thresh, len(speech_buffer))

            if energy > self._silence_thresh:
                speech_buffer.append(chunk)
                silence_start = None
                if buffer_start is None:
                    buffer_start = time.monotonic()
            else:
                if speech_buffer:
                    speech_buffer.append(chunk)
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= self._silence_timeout:
                        logging.info("[AUDIO] Silence detected — transcribing %d chunks", len(speech_buffer))
                        self._transcribe(speech_buffer)
                        speech_buffer = []
                        silence_start = None
                        buffer_start = None

            if buffer_start is not None and time.monotonic() - buffer_start >= self._max_buffer_s:
                logging.info("[AUDIO] Max buffer duration reached — transcribing %d chunks", len(speech_buffer))
                self._transcribe(speech_buffer)
                speech_buffer = []
                silence_start = None
                buffer_start = None

        if speech_buffer:
            self._transcribe(speech_buffer)

    def _transcribe(self, chunks: list[np.ndarray]):
        audio = np.concatenate(chunks)
        if len(audio) < self._sample_rate * 0.3:
            return

        segments, _ = self._model.transcribe(
            audio,
            beam_size=3,
            language="en",
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            logging.info("[AUDIO] Transcribed: %s", text)
            with self._lock:
                if self._latest_text:
                    self._latest_text += " " + text
                else:
                    self._latest_text = text
