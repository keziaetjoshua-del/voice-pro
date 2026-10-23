import argparse
import os
import sys
from pathlib import Path

# Téléchargements Hugging Face : le protocole Xet peut rester bloqué à 0 % sous Windows.
# On force le téléchargement HTTPS classique (mettre HF_HUB_DISABLE_XET=0 pour revenir au défaut).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Windows에서 gradio -> torch -> pyarrow.dataset 순서로 DLL이 로드되면 네이티브 크래시가
# 발생한다 (f5_tts가 datasets를 통해 pyarrow.dataset을 당김). torch와 pyarrow.dataset을
# 먼저 로드해 DLL 해석 순서를 고정한다.
import torch  # noqa: F401
import pyarrow.dataset  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from src.config import UserConfig
from app.abus_hf import AbusHuggingFace
from app.abus_genuine import genuine_init
from app.abus_app_voice import create_ui
from app.abus_path import path_workspace_folder, path_gradio_folder

# ABUS - start voice
genuine_init()
AbusHuggingFace.initialize(app_name="voice")

# AbusHuggingFace.hf_download_models(file_type='mdxnet-model', level=0)
AbusHuggingFace.hf_download_models(file_type='demucs', level=0)
# AbusHuggingFace.hf_download_models(file_type='f5-tts', level=0)
# AbusHuggingFace.hf_download_models(file_type='vocos-mel-24khz', level=0)
# AbusHuggingFace.hf_download_models(file_type='rvc-model', level=0)
# AbusHuggingFace.hf_download_models(file_type='rvc-voice', level=0)
AbusHuggingFace.hf_download_models(file_type='edge-tts', level=0)
AbusHuggingFace.hf_download_models(file_type='kokoro', level=0)
if torch.cuda.is_available() or os.environ.get("VOICEPRO_FULL_DOWNLOAD") == "1":
    AbusHuggingFace.hf_download_models(file_type='cosyvoice', level=0)
else:
    # Sans carte NVIDIA, CosyVoice est beaucoup trop lent : on ne télécharge pas son modèle de 9 Go
    # (seulement les petits échantillons de voix). VOICEPRO_FULL_DOWNLOAD=1 pour tout télécharger.
    print("CPU mode: skipping CosyVoice2-0.5B (9 GB). Set VOICEPRO_FULL_DOWNLOAD=1 to download it.")
    for hf_file in AbusHuggingFace.HF_FILES:
        if hf_file.file_type == 'cosyvoice' and hf_file.level <= 0 \
                and hf_file.file_name != 'CosyVoice2-0.5B.zip' and not hf_file.has_local_file():
            hf_file.download()

path_workspace_folder()
path_gradio_folder()


user_config_path = os.path.join(Path(__file__).resolve().parent, "app", "config-user.json5")
user_config = UserConfig(user_config_path)

create_ui(user_config=user_config)