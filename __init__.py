import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .music_mapper import (
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser
    )
except ImportError:
    from music_mapper import (
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser
    )

NODE_CLASS_MAPPINGS = {
    "GeekatplayAudioToSpectrogram": GeekatplayAudioToSpectrogram,
    "GeekatplaySpectrogramToAudio": GeekatplaySpectrogramToAudio,
    "GeekatplayMusicAnalyser": GeekatplayMusicAnalyser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekatplayAudioToSpectrogram": "GAP Audio To Spectrogram (Geekatplay)",
    "GeekatplaySpectrogramToAudio": "GAP Spectrogram To Audio (Geekatplay)",
    "GeekatplayMusicAnalyser": "GAP Music Analyser & Prompt (Geekatplay)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
