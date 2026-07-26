import os
import sys
import json
import torch
import numpy as np

# Ensure local module directory is in sys.path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Mock folder_paths if running standalone
try:
    import folder_paths
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

from music_mapper import (
    GeekatplayLoadAudio,
    GeekatplayAudioToSpectrogram,
    GeekatplaySpectrogramToAudio,
    GeekatplayMusicAnalyser,
    GeekatplayDisplayTextBox
)

def run_full_pipeline_test():
    print("==========================================================")
    print("      GEEKATPLAY MUSIC MAPPER FULL PIPELINE TEST          ")
    print("==========================================================")
    
    # 1. Synthesize Audio Signal (A Major Chord + Drums)
    sr = 44100
    duration = 5.0 # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Synth notes
    a4 = np.sin(2 * np.pi * 440.0 * t)
    cs5 = np.sin(2 * np.pi * 554.37 * t)
    e5 = np.sin(2 * np.pi * 659.25 * t)
    chord = (a4 + cs5 + e5) / 3.0
    
    # Add rhythmic pulses
    pulse = (np.sin(2 * np.pi * 2.0 * t) > 0.8).astype(float) * 0.3
    audio_signal = chord + pulse
    
    audio_tensor = torch.from_numpy(audio_signal).unsqueeze(0).unsqueeze(0).float()
    audio_input = {
        "waveform": audio_tensor,
        "sample_rate": sr
    }
    
    print(f"\n[STEP 1] Audio Input Created: Shape={audio_tensor.shape}, SR={sr}, Duration={duration}s")
    
    # 2. Run GeekatplayMusicAnalyser (LAION-CLAP Deep Learning Engine)
    print("\n[STEP 2] Running GAP Music Analyser (LAION-CLAP Deep Learning Engine)...")
    analyser_node = GeekatplayMusicAnalyser()
    res = analyser_node.analyze_music(
        audio=audio_input,
        analysis_engine="LAION-CLAP Deep Learning (Auto-Download)",
        ollama_url="http://localhost:11434",
        ollama_model="llama3",
        additional_context="High-fidelity acoustic resonance test"
    )
    if isinstance(res, dict):
        prompt, features_json = res["result"]
    else:
        prompt, features_json = res
    
    print("\n----------------------------------------------------------")
    print("GENERATED MUSICOLOGICAL PROMPT OUTPUT:")
    print("----------------------------------------------------------")
    print(prompt)
    print("----------------------------------------------------------")
    print(f"Prompt Character Length: {len(prompt)}")
    print(f"Features JSON:\n{features_json}")
    
    assert len(prompt) > 30, "ERROR: Generated prompt is too short!"
    assert "BPM" in prompt or "key" in prompt, "ERROR: Missing musical parameters in prompt!"
    print("\n[PASS] Music Analyser generated Suno / Udio style prompt successfully!")
    
    # 3. Test GeekatplayDisplayTextBox UI Output
    print("\n[STEP 3] Testing GAP Display Text Box UI Payload...")
    display_node = GeekatplayDisplayTextBox()
    ui_output = display_node.display_text(text=prompt)
    
    print(f"Display Node UI Dictionary: {ui_output}")
    assert "ui" in ui_output, "ERROR: Display node missing 'ui' key!"
    assert "text" in ui_output["ui"], "ERROR: Display node missing 'text' UI payload!"
    assert "string" in ui_output["ui"], "ERROR: Display node missing 'string' UI payload!"
    assert ui_output["ui"]["text"][0] == prompt, "ERROR: Prompt mismatch in display node!"
    print("[PASS] Display Text Box UI Payload verified successfully!")
    
    # 4. Test GeekatplayAudioToSpectrogram (Phase-Encoded RGB Mode)
    print("\n[STEP 4] Running GAP Audio To Spectrogram (Phase-Encoded RGB Mode)...")
    spec_node = GeekatplayAudioToSpectrogram()
    img_tensor, metadata_json = spec_node.generate_spectrogram(
        audio=audio_input,
        mode="Phase-Encoded RGB (STFT)",
        colormap="Geekatplay Orange Blue",
        n_fft=2048,
        hop_length=512,
        n_mels=512,
        duration=5.0,
        sample_rate=sr,
        channel_mode="mixdown_mono"
    )
    
    print(f"Generated Spectrogram Image Shape: {img_tensor.shape}")
    assert img_tensor.ndim == 4, "ERROR: Spectrogram image must be 4D tensor!"
    assert img_tensor.shape[3] == 3, "ERROR: Image must have 3 RGB channels!"
    print("[PASS] Audio To Spectrogram generated image successfully!")
    
    # 5. Test GeekatplaySpectrogramToAudio (Lossless Reconstruction)
    print("\n[STEP 5] Running GAP Spectrogram To Audio (Lossless STFT Reconstruction)...")
    recon_node = GeekatplaySpectrogramToAudio()
    recon_audio_tuple = recon_node.reconstruct_audio(
        image=img_tensor,
        reconstruct_mode="Auto",
        griffin_lim_iter=32,
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=512,
        metadata_json=metadata_json
    )
    
    recon_audio = recon_audio_tuple[0]
    recon_waveform = recon_audio["waveform"]
    recon_sr = recon_audio["sample_rate"]
    
    print(f"Reconstructed Audio Shape: {recon_waveform.shape}, SR: {recon_sr}")
    assert recon_sr == sr, "ERROR: Reconstructed sample rate mismatch!"
    assert recon_waveform.shape[2] == int(sr * duration), "ERROR: Reconstructed audio length mismatch!"
    
    # Compute correlation
    y_orig_np = audio_signal
    y_recon_np = recon_waveform[0, 0].numpy()
    
    if len(y_recon_np) > len(y_orig_np):
        y_recon_np = y_recon_np[:len(y_orig_np)]
    elif len(y_recon_np) < len(y_orig_np):
        y_recon_np = np.pad(y_recon_np, (0, len(y_orig_np) - len(y_recon_np)))
        
    corr = float(np.corrcoef(y_orig_np, y_recon_np)[0, 1])
    print(f"Audio Reconstruction Signal Correlation: {corr:.6f} (1.0 = Lossless Match)")
    assert corr > 0.95, f"ERROR: Correlation too low: {corr}"
    print("[PASS] Spectrogram To Audio reconstructed audio losslessly!")
    
    print("\n==========================================================")
    print("    ALL FULL PIPELINE TESTS & VALIDATIONS PASSED!          ")
    print("==========================================================")

if __name__ == "__main__":
    run_full_pipeline_test()
