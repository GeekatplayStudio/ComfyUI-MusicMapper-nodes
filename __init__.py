import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from .music_mapper import (
    GeekatplayLoadAudio,
    GeekatplaySaveAudio,
    GeekatplayPreviewAudio,
    GeekatplayDisplayTextBox,
    GeekatplayAudioToSpectrogram,
    GeekatplaySpectrogramToAudio,
    GeekatplayMusicAnalyser
)

NODE_CLASS_MAPPINGS = {
    "GeekatplayLoadAudio": GeekatplayLoadAudio,
    "GeekatplaySaveAudio": GeekatplaySaveAudio,
    "GeekatplayPreviewAudio": GeekatplayPreviewAudio,
    "GeekatplayDisplayTextBox": GeekatplayDisplayTextBox,
    "GeekatplayAudioToSpectrogram": GeekatplayAudioToSpectrogram,
    "GeekatplaySpectrogramToAudio": GeekatplaySpectrogramToAudio,
    "GeekatplayMusicAnalyser": GeekatplayMusicAnalyser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekatplayLoadAudio": "GAP Load Audio (Geekatplay)",
    "GeekatplaySaveAudio": "GAP Save Audio (Geekatplay)",
    "GeekatplayPreviewAudio": "GAP Preview Audio (Geekatplay)",
    "GeekatplayDisplayTextBox": "GAP Display Text Box (Geekatplay)",
    "GeekatplayAudioToSpectrogram": "GAP Audio To Spectrogram (Geekatplay)",
    "GeekatplaySpectrogramToAudio": "GAP Spectrogram To Audio (Geekatplay)",
    "GeekatplayMusicAnalyser": "GAP Music Analyser & Prompt (Geekatplay)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
