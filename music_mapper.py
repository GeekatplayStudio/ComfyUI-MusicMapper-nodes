import os
import json
import math
import numpy as np
import torch
import soundfile as sf
import librosa
import requests
from PIL import Image
import matplotlib.pyplot as plt
import folder_paths

# Setup default configurations
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 512
EPSILON = 1e-8

def estimate_key(y, sr):
    """
    Estimates the musical key and scale of the audio using the Krumhansl-Schmuckler profile correlation.
    """
    try:
        # If stereo, mixdown to mono
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Normalize profiles (zero-mean, unit-variance)
        major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
        minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)

        best_corr = -999.0
        best_key = "C Major"

        for shift in range(12):
            shifted_chroma = np.roll(chroma_mean, -shift)
            
            # Normalize shifted chroma
            if np.std(shifted_chroma) > 0:
                shifted_chroma = (shifted_chroma - np.mean(shifted_chroma)) / np.std(shifted_chroma)
            else:
                shifted_chroma = shifted_chroma - np.mean(shifted_chroma)

            # Major correlation
            corr_maj = np.dot(shifted_chroma, major_profile)
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key = f"{notes[shift]} Major"

            # Minor correlation
            corr_min = np.dot(shifted_chroma, minor_profile)
            if corr_min > best_corr:
                best_corr = corr_min
                best_key = f"{notes[shift]} Minor"

        return best_key
    except Exception as e:
        print(f"[Geekatplay MusicMapper] Key estimation error: {e}")
        return "Unknown Key"

def apply_colormap(data_norm, colormap_name):
    """
    Applies a color mapping to a 2D normalized [0, 1] array, returning [H, W, 3] RGB array in range [0, 1].
    """
    if colormap_name == "Grayscale":
        return np.stack([data_norm, data_norm, data_norm], axis=-1)
    
    elif colormap_name == "Geekatplay Orange Blue":
        # Multi-stage custom linear interpolation
        # 0.0: Dark Blue (#03071e)
        # 0.25: Royal Blue (#0d47a1)
        # 0.5: Deep Violet/Purple (#7b1fa2)
        # 0.75: Bright Orange (#ff6d00)
        # 1.0: Warm Gold/Yellow (#ffeb3b)
        colors = np.array([
            [3, 7, 30],
            [13, 71, 161],
            [123, 31, 162],
            [255, 109, 0],
            [255, 235, 59]
        ]) / 255.0
        
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r = np.interp(data_norm, x, colors[:, 0])
        g = np.interp(data_norm, x, colors[:, 1])
        b = np.interp(data_norm, x, colors[:, 2])
        return np.stack([r, g, b], axis=-1)
    
    else:
        try:
            cmap = plt.get_cmap(colormap_name.lower())
            rgba = cmap(data_norm)
            return rgba[..., :3]  # Drop alpha channel
        except Exception as e:
            print(f"[Geekatplay MusicMapper] Error applying colormap '{colormap_name}': {e}. Falling back to Grayscale.")
            return np.stack([data_norm, data_norm, data_norm], axis=-1)

def invert_colormap_to_grayscale(rgb_img):
    """
    Converts RGB image back to grayscale using luminance standard (ITU-R BT.601).
    This works perfectly for monotonically increasing colormaps (Grayscale, Viridis, Plasma, Magma, Geekatplay).
    """
    # RGB shape [H, W, 3] -> return [H, W]
    r, g, b = rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


class GeekatplayLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        return {
            "required": {
                "audio_file": (sorted(files) if files else [""], {"audio_upload": True}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "load_audio"
    CATEGORY = "Geekatplay Studio/Audio"

    def load_audio(self, audio_file="", custom_path=""):
        resolved_path = None
        
        # 1. Prioritize custom_path if provided and exists
        if custom_path and custom_path.strip():
            clean_path = custom_path.strip().strip('"').strip("'")
            if os.path.isfile(clean_path):
                resolved_path = clean_path
            else:
                annotated = folder_paths.get_annotated_filepath(clean_path)
                if os.path.isfile(annotated):
                    resolved_path = annotated

        # 2. Fall back to audio_file (from ComfyUI upload/input folder)
        if not resolved_path and audio_file and audio_file.strip():
            annotated = folder_paths.get_annotated_filepath(audio_file)
            if os.path.isfile(annotated):
                resolved_path = annotated
            else:
                input_dir = folder_paths.get_input_directory()
                joined = os.path.join(input_dir, audio_file.strip())
                if os.path.isfile(joined):
                    resolved_path = joined

        if not resolved_path or not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"Audio file not found. Custom path: '{custom_path}', Audio file input: '{audio_file}'. "
                "Please upload an audio file or enter a valid file path."
            )

        # Load with librosa to support various formats and get float waveform
        y, sr = librosa.load(resolved_path, sr=None, mono=False)
        
        # Format to ComfyUI AUDIO standard: {"waveform": torch.Tensor [1, channels, samples], "sample_rate": int}
        if y.ndim == 1:
            # Mono
            waveform = torch.from_numpy(y).unsqueeze(0).unsqueeze(0) # [1, 1, samples]
        else:
            # Stereo/Multichannel
            waveform = torch.from_numpy(y).unsqueeze(0) # [1, channels, samples]

        audio = {
            "waveform": waveform.float(),
            "sample_rate": int(sr)
        }
        return (audio, resolved_path)


class GeekatplaySaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "Geekatplay_Audio"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_audio"
    CATEGORY = "Geekatplay Studio/Audio"
    OUTPUT_NODE = True

    def save_audio(self, audio, filename_prefix):
        waveform = audio["waveform"] # [1, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # Select first batch item
        y_tensor = waveform[0].cpu()
        channels = y_tensor.shape[0]
        
        # soundfile expects (samples, channels)
        y_np = y_tensor.numpy().T
        
        output_dir = folder_paths.get_output_directory()
        
        # Generate unique filename
        import uuid
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.wav"
        save_path = os.path.join(output_dir, filename)
        
        sf.write(save_path, y_np, sample_rate)
        print(f"[Geekatplay MusicMapper] Saved audio file to: {save_path}")
        return (save_path,)


class GeekatplayAudioToSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["Mel-Spectrogram (Standard Training)", "Phase-Encoded RGB (STFT)"], {"default": "Mel-Spectrogram (Standard Training)"}),
                "colormap": (["Grayscale", "Geekatplay Orange Blue", "Viridis", "Plasma", "Magma", "Inferno"], {"default": "Geekatplay Orange Blue"}),
                "n_fft": ("INT", {"default": DEFAULT_N_FFT, "min": 256, "max": 8192, "step": 256}),
                "hop_length": ("INT", {"default": DEFAULT_HOP_LENGTH, "min": 64, "max": 4096, "step": 64}),
                "n_mels": ("INT", {"default": DEFAULT_N_MELS, "min": 64, "max": 1024, "step": 64}),
                "duration": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 300.0, "step": 0.5}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000}),
                "channel_mode": (["mixdown_mono", "stereo_vertical", "left_only", "right_only"], {"default": "mixdown_mono"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "generate_spectrogram"
    CATEGORY = "Geekatplay Studio/Spectrogram"

    def generate_spectrogram(self, audio, mode, colormap, n_fft, hop_length, n_mels, duration, sample_rate, channel_mode):
        waveform = audio["waveform"][0].cpu().numpy() # shape [channels, samples]
        orig_sr = audio["sample_rate"]
        
        # Resample if needed
        if orig_sr != sample_rate:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sample_rate)
            
        # Select/mix channels
        if channel_mode == "mixdown_mono":
            y = np.mean(waveform, axis=0, keepdims=True)
        elif channel_mode == "left_only":
            y = waveform[0:1]
        elif channel_mode == "right_only":
            y = waveform[1:2] if waveform.shape[0] > 1 else waveform[0:1]
        else: # stereo_vertical
            y = waveform[0:2] # Force stereo limit

        # Trim or pad to duration
        num_channels = y.shape[0]
        if duration > 0.0:
            target_samples = int(duration * sample_rate)
            if y.shape[-1] < target_samples:
                # Pad with zeros
                pad_width = target_samples - y.shape[-1]
                y = np.pad(y, ((0, 0), (0, pad_width)), mode='constant')
            elif y.shape[-1] > target_samples:
                # Crop
                y = y[:, :target_samples]

        panels = []
        metadata_channels = []
        
        for c in range(num_channels):
            channel_y = y[c]
            
            if mode == "Mel-Spectrogram (Standard Training)":
                # Compute Mel spectrogram
                S = librosa.feature.melspectrogram(
                    y=channel_y, 
                    sr=sample_rate, 
                    n_fft=n_fft, 
                    hop_length=hop_length, 
                    n_mels=n_mels
                )
                S_db = librosa.power_to_db(S, ref=1.0)
                
                # Scale DB to [0, 1]
                db_min = float(S_db.min())
                db_max = float(S_db.max())
                db_span = db_max - db_min
                if db_span < EPSILON:
                    db_span = EPSILON
                
                S_norm = (S_db - db_min) / db_span
                
                # Apply colormap
                panel = apply_colormap(S_norm, colormap)
                
                # We need to flip vertical so that low frequencies are at the bottom!
                panel = np.flipud(panel)
                panels.append(panel)
                
                metadata_channels.append({
                    "channel_index": c,
                    "db_min": db_min,
                    "db_max": db_max
                })
                
            else: # Phase-Encoded RGB (STFT)
                # Compute STFT
                D = librosa.stft(
                    y=channel_y,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                magnitude = np.abs(D)
                phase = np.angle(D)
                
                log_magnitude = np.log1p(magnitude)
                mag_min = float(log_magnitude.min())
                mag_max = float(log_magnitude.max())
                mag_span = mag_max - mag_min
                if mag_span < EPSILON:
                    mag_span = EPSILON
                
                mag_norm = (log_magnitude - mag_min) / mag_span
                
                # Phase encoding in G and B channels
                phase_cos = (np.cos(phase) + 1.0) * 0.5
                phase_sin = (np.sin(phase) + 1.0) * 0.5
                
                # Combine to RGB: R=magnitude, G=cos_phase, B=sin_phase
                panel = np.stack([mag_norm, phase_cos, phase_sin], axis=-1)
                
                # Flip vertical
                panel = np.flipud(panel)
                panels.append(panel)
                
                metadata_channels.append({
                    "channel_index": c,
                    "mag_min": mag_min,
                    "mag_max": mag_max
                })

        # Stack channels vertically if stereo_vertical, otherwise panels list is length 1
        final_img = np.vstack(panels)

        # Convert to ComfyUI standard: PyTorch float tensor [batch, height, width, channels]
        image_tensor = torch.from_numpy(final_img).unsqueeze(0).float()

        # Build metadata JSON
        meta = {
            "brand": "Geekatplay Studio",
            "creator": "Vladimir Chopine",
            "mode": mode,
            "colormap": colormap,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "duration": duration,
            "channels": num_channels,
            "original_samples": int(y.shape[-1]),
            "channel_metadata": metadata_channels
        }
        
        metadata_json = json.dumps(meta, indent=2)
        return (image_tensor, metadata_json)


class GeekatplaySpectrogramToAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reconstruct_mode": (["Auto", "Mel-Spectrogram Griffin-Lim", "Phase-Encoded RGB"], {"default": "Auto"}),
                "griffin_lim_iter": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000}),
                "n_fft": ("INT", {"default": DEFAULT_N_FFT, "min": 256, "max": 8192, "step": 256}),
                "hop_length": ("INT", {"default": DEFAULT_HOP_LENGTH, "min": 64, "max": 4096, "step": 64}),
                "n_mels": ("INT", {"default": DEFAULT_N_MELS, "min": 64, "max": 1024, "step": 64}),
            },
            "optional": {
                "metadata_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "reconstruct_audio"
    CATEGORY = "Geekatplay Studio/Spectrogram"

    def reconstruct_audio(self, image, reconstruct_mode, griffin_lim_iter, sample_rate, n_fft, hop_length, n_mels, metadata_json=""):
        # Parse metadata if present
        meta = {}
        if metadata_json and metadata_json.strip():
            try:
                meta = json.loads(metadata_json)
            except Exception as e:
                print(f"[Geekatplay MusicMapper] Error parsing metadata: {e}")

        # Resolve parameters from metadata if Auto, or use provided values
        mode = meta.get("mode", "Mel-Spectrogram (Standard Training)")
        if reconstruct_mode != "Auto":
            if reconstruct_mode == "Mel-Spectrogram Griffin-Lim":
                mode = "Mel-Spectrogram (Standard Training)"
            else:
                mode = "Phase-Encoded RGB (STFT)"

        sr = meta.get("sample_rate", sample_rate)
        fft_len = meta.get("n_fft", n_fft)
        hop = meta.get("hop_length", hop_length)
        mels = meta.get("n_mels", n_mels)
        channels = meta.get("channels", 1)
        original_samples = meta.get("original_samples", 0)
        channel_metadata = meta.get("channel_metadata", [])

        # image is [batch, height, width, channels]
        img_np = image[0].cpu().numpy() # [H, W, C]
        
        # Split image vertically if multiple channels
        panel_height = img_np.shape[0] // channels
        
        reconstructed_waveforms = []
        
        for c in range(channels):
            # Extract panel
            start_y = c * panel_height
            end_y = start_y + panel_height
            panel = img_np[start_y:end_y]
            
            # Flip vertical back to normal spectrogram orientation (reversing the flipud done on save)
            panel = np.flipud(panel)
            
            # Get channel-specific min/max scaling
            c_meta = channel_metadata[c] if c < len(channel_metadata) else {}
            
            if mode == "Mel-Spectrogram (Standard Training)":
                db_min = c_meta.get("db_min", -80.0)
                db_max = c_meta.get("db_max", 0.0)
                
                # Invert colormap to normalized grayscale [0, 1]
                S_norm = invert_colormap_to_grayscale(panel)
                
                # Rescale to dB
                S_db = S_norm * (db_max - db_min) + db_min
                
                # Convert dB back to power
                S = librosa.db_to_power(S_db)
                
                # Run Griffin-Lim reconstruction from Mel-spectrogram
                y_recon = librosa.feature.inverse.mel_to_audio(
                    S,
                    sr=sr,
                    n_fft=fft_len,
                    hop_length=hop,
                    n_iter=griffin_lim_iter
                )
                
            else: # Phase-Encoded RGB (STFT)
                mag_min = c_meta.get("mag_min", 0.0)
                mag_max = c_meta.get("mag_max", 10.0)
                
                # Decode magnitude from Red channel
                mag_norm = panel[..., 0]
                log_magnitude = mag_norm * (mag_max - mag_min) + mag_min
                magnitude = np.expm1(log_magnitude)
                
                # Decode phase from Green and Blue channels
                phase_cos = panel[..., 1] * 2.0 - 1.0
                phase_sin = panel[..., 2] * 2.0 - 1.0
                phase = np.arctan2(phase_sin, phase_cos)
                
                # Reconstruct complex matrix
                D = magnitude * np.exp(1j * phase)
                
                # Inverse STFT
                y_recon = librosa.istft(
                    D,
                    hop_length=hop,
                    win_length=fft_len
                )

            # Crop or pad to original sample size if known
            if original_samples > 0:
                if len(y_recon) > original_samples:
                    y_recon = y_recon[:original_samples]
                elif len(y_recon) < original_samples:
                    y_recon = np.pad(y_recon, (0, original_samples - len(y_recon)), mode='constant')
                
            reconstructed_waveforms.append(y_recon)
            
        # Stack waveforms into [1, channels, samples]
        waveform_np = np.stack(reconstructed_waveforms, axis=0)
        
        # Audio output
        audio = {
            "waveform": torch.from_numpy(waveform_np).unsqueeze(0).float(),
            "sample_rate": int(sr)
        }
        return (audio,)


class GeekatplayMusicAnalyser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "use_ollama": ("BOOLEAN", {"default": True}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434"}),
                "ollama_model": ("STRING", {"default": "llama3"}),
            },
            "optional": {
                "additional_context": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "extracted_features_json")
    FUNCTION = "analyze_music"
    CATEGORY = "Geekatplay Studio/Audio"

    def analyze_music(self, audio, use_ollama, ollama_url, ollama_model, additional_context=""):
        waveform = audio["waveform"][0].cpu().numpy() # shape [channels, samples]
        sr = audio["sample_rate"]
        
        # Mixdown to mono for analysis
        y = np.mean(waveform, axis=0)
        
        # 1. Estimate BPM / Tempo
        print("[Geekatplay MusicMapper] Estimating Tempo...")
        try:
            tempo_data = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo_data, tuple):
                tempo = tempo_data[0]
            else:
                tempo = tempo_data
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if tempo.size > 0 else 120.0
            else:
                tempo = float(tempo)
        except Exception as e:
            print(f"[Geekatplay MusicMapper] Tempo detection failed: {e}")
            tempo = 120.0
            
        # 2. Key detection
        print("[Geekatplay MusicMapper] Estimating Key and Scale...")
        detected_key = estimate_key(y, sr)
        
        # 3. DSP Spectral and Dynamics feature extraction
        print("[Geekatplay MusicMapper] Performing Spectral Feature Extraction...")
        # Spectral Centroid (relates to perceived brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(centroid))
        
        # Spectral Rolloff (relates to high-frequency concentration)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        # Zero Crossing Rate (relates to noisiness/sibilance/percussion)
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))
        
        # RMS Energy (relates to volume/energy)
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        dynamic_range = std_rms / (mean_rms + EPSILON)
        
        # Map values to descriptive acoustic terms
        # Tempo description
        if tempo < 75:
            tempo_desc = "Slow, ambient, lingering tempo with extended sustained waveforms"
            rhythm_pattern = "sluggishly evolving rhythmic cycles and long-form sound waves"
        elif tempo < 100:
            tempo_desc = "Relaxed, moderate, steady pulse with laid-back rhythmic groove"
            rhythm_pattern = "gently rolling beats, smooth cadences, and steady meters"
        elif tempo < 120:
            tempo_desc = "Walking moderate tempo with active rhythmic pulse"
            rhythm_pattern = "structured steady drive, clear bar measure pulses, and regular cadence"
        elif tempo < 140:
            tempo_desc = "Upbeat, driving, high-tempo energy with prominent metric pulses"
            rhythm_pattern = "rapid metric subdivisions, driving syncopated accents, and energetic pulse"
        else:
            tempo_desc = "Fast-paced, hyper-velocity rush with dense rhythmic density"
            rhythm_pattern = "rapid transient spikes, kinetic rhythmic density, and high-frequency impulses"

        # Key Scale description (Major vs Minor)
        is_minor = "minor" in detected_key.lower()
        if is_minor:
            tonality_desc = "introspective, emotionally somber, minor-key harmonic structure"
            consonance_desc = "complex harmonic tension, dark modal undertones, and brooding chord progressions"
        else:
            tonality_desc = "bright, uplifting, major-key harmonic structure"
            consonance_desc = "resonant consonant harmonies, stable tonal centers, and triumphant chord voicings"

        # Brightness (Centroid) description
        if mean_centroid < 1200:
            brightness_desc = "deep, sub-bass heavy, warm timbral profile with low spectral centroid"
            freq_balance = "heavy fundamental bass energy, muted upper harmonics, and dark warm timbre"
        elif mean_centroid < 2400:
            brightness_desc = "balanced mid-range timbral profile with natural spectral distribution"
            freq_balance = "well-defined vocal and instrumental mid-range frequencies with organic acoustic balance"
        else:
            brightness_desc = "sparkling, bright, treble-dominant timbral profile with high spectral centroid"
            freq_balance = "crisp high-frequency overtones, sharp metallic sibilance, and brilliant presence"

        # Zero Crossing Rate (Noise/Percussiveness) description
        if mean_zcr < 0.04:
            timbre_desc = "purely tonal, smooth sinusoidal harmonics, and clean melodic contours"
            percussion_desc = "minimal transient noise, dominated by legato melodic lines and pure acoustic tones"
        elif mean_zcr < 0.12:
            timbre_desc = "balanced acoustic texture with mixed tonal and percussive transients"
            percussion_desc = "defined drum hits, articulate articulation, and balanced harmonic noise"
        else:
            timbre_desc = "highly percussive, noisy, distorted, or heavily transient acoustic profile"
            percussion_desc = "sharp attack transients, dense noise bursts, complex percussive articulation, and static grain"

        # RMS Energy description
        if mean_rms < 0.015:
            dynamic_desc = "whisper-soft, delicate dynamic profile with low overall loudness"
            amplitude_envelope = "subtle low-amplitude fluctuations and quiet intimate dynamics"
        elif mean_rms < 0.08:
            dynamic_desc = "moderate presence with consistent controlled volume"
            amplitude_envelope = "steady dynamic headroom, balanced sound pressure level, and readable contrast"
        else:
            dynamic_desc = "intensely loud, high-energy, wall-of-sound dynamic compression"
            amplitude_envelope = "maximum RMS energy density, peak sound pressure level, and explosive volume peaks"

        features_meta = {
            "estimated_tempo_bpm": round(tempo, 1),
            "estimated_key": detected_key,
            "spectral_centroid_hz": round(mean_centroid, 1),
            "spectral_rolloff_hz": round(mean_rolloff, 1),
            "zero_crossing_rate": round(mean_zcr, 4),
            "rms_energy_mean": round(mean_rms, 4),
            "rms_energy_std": round(std_rms, 4),
            "dynamic_range_ratio": round(dynamic_range, 4),
            "tonality_type": "Minor / Introspective" if is_minor else "Major / Uplifting",
            "brightness_profile": "Dark / Sub-bass" if mean_centroid < 1200 else ("Bright / Treble" if mean_centroid > 2400 else "Balanced / Mid-range")
        }
        features_json = json.dumps(features_meta, indent=2)

        # Pure Musicological Analysis Prompt
        fallback_prompt = (
            f"Geekatplay Studio Audio Analysis Report by Vladimir Chopine. "
            f"Detailed Acoustic & Musicological Profile: The composition is set to an estimated tempo of {tempo:.1f} BPM ({tempo_desc}), characterized by {rhythm_pattern}. "
            f"Harmonically, the track is in the key of {detected_key} ({tonality_desc}), exhibiting {consonance_desc}. "
            f"The spectral centroid averages {mean_centroid:.1f} Hz ({brightness_desc}), resulting in {freq_balance}. "
            f"Timbrally, the signal exhibits a zero crossing rate of {mean_zcr:.4f} ({timbre_desc}), featuring {percussion_desc}. "
            f"With a mean RMS energy level of {mean_rms:.4f} ({dynamic_desc}), the audio displays an amplitude envelope with {amplitude_envelope}. "
            f"Spectral rolloff cutoff is at {mean_rolloff:.1f} Hz with a dynamic range ratio of {dynamic_range:.4f}."
        )
        
        if additional_context and additional_context.strip():
            fallback_prompt += f" Additional Context: {additional_context.strip()}."

        if len(fallback_prompt) < 950:
            expansion = (
                f" This detailed audio feature signature accurately describes the physical sound waves, tonal key center, "
                f"frequency distribution, transient density, and dynamic loudness of the recording for model mapping and acoustic synthesis."
            )
            fallback_prompt += expansion

        final_prompt = fallback_prompt

        # 2. Ollama Prompt Generator
        if use_ollama:
            print(f"[Geekatplay MusicMapper] Querying local Ollama model '{ollama_model}' at {ollama_url}...")
            
            system_instruction = (
                "You are an expert musicologist and audio signal analyst for Geekatplay Studio by Vladimir Chopine. "
                "Your task is to write an extensive, highly detailed, professional musicological description (around 1000 characters) analyzing the audio's musical characteristics, key, tempo, acoustic dynamics, timbre, frequency balance, and rhythm. "
                "Do NOT include any visual art styles (no paintings, no cyberpunks, no surrealism). Focus 100% purely on the music and sound wave analysis. Output ONLY the raw description paragraph."
            )
            
            prompt_input = (
                f"Write a comprehensive music analysis paragraph (around 1000 characters) based on these extracted DSP audio features:\n"
                f"- Tempo: {tempo:.1f} BPM ({tempo_desc})\n"
                f"- Rhythmic Pattern: {rhythm_pattern}\n"
                f"- Key & Tonality: {detected_key} ({tonality_desc})\n"
                f"- Harmonic Tension: {consonance_desc}\n"
                f"- Frequency Centroid: {mean_centroid:.1f} Hz ({brightness_desc})\n"
                f"- Timbre & Noise (Zero Crossing): {mean_zcr:.4f} ({timbre_desc})\n"
                f"- Dynamic Energy (RMS): {mean_rms:.4f} ({dynamic_desc})\n"
                f"- Additional Context: {additional_context if additional_context else 'None'}\n\n"
                f"Write a continuous, highly detailed, academic yet accessible analysis of this music. Include Geekatplay Studio and Vladimir Chopine."
            )
            
            try:
                # Prepare Ollama API payload
                url = f"{ollama_url.rstrip('/')}/api/generate"
                payload = {
                    "model": ollama_model,
                    "prompt": f"{system_instruction}\n\nUser Request:\n{prompt_input}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 350
                    }
                }
                
                # Make HTTP request to local Ollama with a 6-second timeout
                response = requests.post(url, json=payload, timeout=6.0)
                if response.status_code == 200:
                    resp_json = response.json()
                    response_text = resp_json.get("response", "")
                    if not response_text and "message" in resp_json:
                        response_text = resp_json["message"].get("content", "")
                    response_text = response_text.strip()
                    
                    if len(response_text) > 100:
                        final_prompt = response_text
                        print("[Geekatplay MusicMapper] Ollama prompt generated successfully.")
                    else:
                        print(f"[Geekatplay MusicMapper] Ollama response too short ({len(response_text)} chars). Using rule-based fallback.")
                else:
                    print(f"[Geekatplay MusicMapper] Ollama returned status {response.status_code}. Using rule-based fallback.")
            except requests.exceptions.Timeout:
                print("[Geekatplay MusicMapper] Ollama connection timed out. Using rule-based fallback.")
            except Exception as e:
                print(f"[Geekatplay MusicMapper] Ollama error: {e}. Using rule-based fallback.")

        # Ultimate safety check: ensure final_prompt is never empty
        if not final_prompt or len(final_prompt.strip()) < 50:
            print("[Geekatplay MusicMapper] Prompt was empty. Using rule-based fallback engine.")
            final_prompt = fallback_prompt

        return (final_prompt, features_json)


class GeekatplayDisplayTextBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "display_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_text"
    CATEGORY = "Geekatplay Studio/Utility"
    
    OUTPUT_NODE = True

    def display_text(self, text, display_text=""):
        # Print to console log
        print(f"\n--- Geekatplay Studio Music Description ---\n{text}\n--------------------------------------------\n")
        # Return UI event for web frontend rendering and STRING output for piping into SD nodes
        return {"ui": {"string": [text]}, "result": (text,)}
