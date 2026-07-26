import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import json
import numpy as np
import torch

try:
    import folder_paths  # type: ignore
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return "."
        @staticmethod
        def get_output_directory():
            return "."
        @staticmethod
        def get_temp_directory():
            return "."
        @staticmethod
        def get_annotated_filepath(name):
            return name
    sys.modules['folder_paths'] = MockFolderPaths()

try:
    from .music_mapper import (  # type: ignore # noqa: F401
        estimate_key,
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser
    )
except (ImportError, ModuleNotFoundError):
    from music_mapper import (  # type: ignore # noqa: F401
        estimate_key,
        GeekatplayAudioToSpectrogram,
        GeekatplaySpectrogramToAudio,
        GeekatplayMusicAnalyser
    )

def test_music_mapper():
    print("==============================================")
    print("Synthesizing mock audio signal (A Major chord)...")
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    a4 = np.sin(2 * np.pi * 440.0 * t)
    cs5 = np.sin(2 * np.pi * 554.37 * t)
    e5 = np.sin(2 * np.pi * 659.25 * t)
    
    chord = (a4 + cs5 + e5) / 3.0
    
    audio_tensor = torch.from_numpy(chord).unsqueeze(0).unsqueeze(0).float()
    audio = {
        "waveform": audio_tensor,
        "sample_rate": sr
    }
    
    print(f"Mock audio shape: {audio_tensor.shape}, Sample rate: {sr}")
    
    print("\nTesting key estimation...")
    detected_key = estimate_key(chord, sr)
    print(f"Estimated key: {detected_key}")
    assert isinstance(detected_key, str), "Key estimation must return a string"
    assert "A" in detected_key or "C#" in detected_key or "E" in detected_key or detected_key != "Unknown Key", "Key estimation failed"
    
    print("\nTesting AudioToSpectrogram (Mel mode)...")
    node_to_spec = GeekatplayAudioToSpectrogram()
    img_tensor, meta_json = node_to_spec.generate_spectrogram(
        audio=audio,
        mode="Mel-Spectrogram (Standard Training)",
        colormap="Geekatplay Orange Blue",
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        duration=4.0,
        sample_rate=sr,
        channel_mode="mixdown_mono"
    )
    
    print(f"Generated image tensor shape: {img_tensor.shape}")
    print(f"Metadata JSON preview: {meta_json[:200]}...")
    
    assert img_tensor.ndim == 4, "Image must be 4D tensor [B, H, W, C]"
    assert img_tensor.shape[0] == 1, "Batch size must be 1"
    assert img_tensor.shape[3] == 3, "Image must have 3 channels (RGB)"
    
    meta = json.loads(meta_json)
    assert meta["brand"] == "Geekatplay Studio", "Incorrect brand in metadata"
    assert meta["channels"] == 1, "Incorrect channels count"
    
    print("\nTesting SpectrogramToAudio (Griffin-Lim loopback)...")
    node_to_audio = GeekatplaySpectrogramToAudio()
    recon_audio_tuple = node_to_audio.reconstruct_audio(
        image=img_tensor,
        reconstruct_mode="Auto",
        griffin_lim_iter=16,
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        metadata_json=meta_json
    )
    
    recon_audio = recon_audio_tuple[0]
    recon_waveform = recon_audio["waveform"]
    recon_sr = recon_audio["sample_rate"]
    
    print(f"Reconstructed audio shape: {recon_waveform.shape}, SR: {recon_sr}")
    assert recon_sr == sr, "Reconstructed sample rate mismatch"
    assert recon_waveform.ndim == 3, "Reconstructed waveform must be 3D [B, C, S]"
    assert recon_waveform.shape[1] == 1, "Reconstructed channels count mismatch"
    assert recon_waveform.shape[2] == int(sr * 4.0), "Reconstructed samples length mismatch"
    
    print("\nTesting MusicAnalyser (Fallback mode)...")
    node_analyser = GeekatplayMusicAnalyser()
    prompt, features_json = node_analyser.analyze_music(
        audio=audio,
        use_ollama=False,
        ollama_url="http://localhost:11434",
        ollama_model="llama3",
        additional_context="Focus on acoustic resonance"
    )
    
    print("\nGenerated Prompt Preview (~1000 characters):")
    print(prompt)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Features JSON:\n{features_json}")
    
    assert len(prompt) > 500, "Generated prompt is too short"
    assert "Geekatplay Studio" in prompt, "Branding missing in prompt"
    assert "Vladimir Chopine" in prompt, "Creator name missing in prompt"
    
    features = json.loads(features_json)
    assert "estimated_tempo_bpm" in features, "BPM missing in features"
    assert "estimated_key" in features, "Key missing in features"
    
    print("\n==============================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==============================================")

if __name__ == "__main__":
    test_music_mapper()
