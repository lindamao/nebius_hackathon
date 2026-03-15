import os
from dotenv import load_dotenv

load_dotenv()

# --- Nebius LLM ---
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_MODEL = "openai/gpt-oss-120b"

# --- LLM Throttling ---
LLM_MIN_INTERVAL_S = 3.0
LLM_IDLE_TIMEOUT_S = 8.0
EXPRESSION_CHANGE_THRESHOLD = 0.3
POSE_CHANGE_THRESHOLD = 0.15

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# --- Audio / STT ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK_DURATION_S = 0.5
SILENCE_THRESHOLD = 0.01
SILENCE_TIMEOUT_S = 1.0
MAX_SPEECH_BUFFER_S = 10.0
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"

# --- Robot Actions ---
ACTIONS = ["cheerful_dance", "hug", "blow_kisses", "kungfu_fighting", "nothing"]
