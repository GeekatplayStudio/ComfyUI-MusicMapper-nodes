import os
import sys

# Ensure local module directory is in sys.path for IDE resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .music_mapper import (
        GeekatplayLoadAudio,
        GeekatplaySaveAudio,
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser,
        GeekatplayDisplayTextBox
    )
except ImportError:
    from music_mapper import (
        GeekatplayLoadAudio,
        GeekatplaySaveAudio,
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser,
        GeekatplayDisplayTextBox
    )

NODE_CLASS_MAPPINGS = {
    "GeekatplayLoadAudio": GeekatplayLoadAudio,
    "GeekatplaySaveAudio": GeekatplaySaveAudio,
    "GeekatplayAudioToSpectrogram": GeekatplayAudioToSpectrogram,
    "GeekatplaySpectrogramToAudio": GeekatplaySpectrogramToAudio,
    "GeekatplayMusicAnalyser": GeekatplayMusicAnalyser,
    "GeekatplayDisplayTextBox": GeekatplayDisplayTextBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekatplayLoadAudio": "GAP Load Audio (Geekatplay)",
    "GeekatplaySaveAudio": "GAP Save Audio (Geekatplay)",
    "GeekatplayAudioToSpectrogram": "GAP Audio To Spectrogram (Geekatplay)",
    "GeekatplaySpectrogramToAudio": "GAP Spectrogram To Audio (Geekatplay)",
    "GeekatplayMusicAnalyser": "GAP Music Analyser & Prompt (Geekatplay)",
    "GeekatplayDisplayTextBox": "GAP Display Text Box (Geekatplay)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
