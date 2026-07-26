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
                "art_style": (["Abstract Expressionism", "Surrealism", "Synthwave / Cyberpunk", "Cosmic / Nebula", "Fluid Dynamics", "Orchestral Cinematic"], {"default": "Cosmic / Nebula"}),
                "additional_context": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "extracted_features_json")
    FUNCTION = "analyze_music"
    CATEGORY = "Geekatplay Studio/Audio"

    def analyze_music(self, audio, use_ollama, ollama_url, ollama_model, art_style, additional_context):
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
        
        # Map values to descriptive terms for prompt construction
        # Tempo description
        if tempo < 75:
            tempo_desc = "Slow, meditative, ambient, and lingering pace"
            movement_desc = "sluggishly flowing currents, slow-motion shifts, wispy floating trails"
        elif tempo < 100:
            tempo_desc = "Relaxed, chill, and steady pulse"
            movement_desc = "gently rolling waves, smooth rhythmic swaying, steady steps"
        elif tempo < 120:
            tempo_desc = "Moderate tempo, active and walking groove"
            movement_desc = "flowing lines, moderately moving geometric streams, rhythmic pulses"
        elif tempo < 140:
            tempo_desc = "Upbeat, driving, and energetic dance tempo"
            movement_desc = "vibrant rapid pulses, structured grid expansions, jumping particles"
        else:
            tempo_desc = "Fast-paced, hyperactive, high-velocity rush"
            movement_desc = "lightning-fast sparks, kinetic streaks of light, chaotic explosive energy"

        # Key Scale description (Major vs Minor)
        is_minor = "minor" in detected_key.lower()
        if is_minor:
            mood_desc = "introspective, melancholic, serious, dramatic, and emotionally deep"
            color_desc = "cool tones, deep indigos, charcoals, shadowed hues, and obsidian blacks"
        else:
            mood_desc = "uplifting, triumphant, bright, positive, joyous, and harmonious"
            color_desc = "warm tones, radiant golds, brilliant ambers, sunlit yellow, and bright creams"

        # Brightness (Centroid) description
        if mean_centroid < 1200:
            brightness_desc = "deep, warm, muddy, bass-heavy, dark, and sub-surface"
            texture_desc = "velvety shadows, thick heavy smoke, solid obsidian surfaces"
        elif mean_centroid < 2400:
            brightness_desc = "balanced, mid-range, natural, organic, and resonant"
            texture_desc = "tactile wood, earthy grains, woven organic fibers"
        else:
            brightness_desc = "extremely bright, sparkling, sharp, metallic, high-frequency, and clinical"
            texture_desc = "fractured crystal glass, brilliant glass reflections, sharp neon edges"

        # Zero Crossing Rate (Noise/Percussiveness) description
        if mean_zcr < 0.04:
            noise_desc = "purely melodic, clean tones, smooth harmonics, and liquid-like transitions"
            form_desc = "perfect curves, continuous uninterrupted sweeps, fluid spheres"
        elif mean_zcr < 0.12:
            noise_desc = "standard acoustic textures, blended vocals, and balanced percussion"
            form_desc = "combination of rounded corners and gentle hatch lines"
        else:
            noise_desc = "harsh, percussive, distorted, white noise, and intensely textured"
            form_desc = "spiky crystalline matrices, static noise grain, jagged fractures"

        # RMS Energy description
        if mean_rms < 0.015:
            energy_desc = "whisper-soft, delicate, silent, and fragile"
            contrast_desc = "subtle low-contrast textures, soft blending edges"
        elif mean_rms < 0.08:
            energy_desc = "moderate presence, structured and stable volume"
            contrast_desc = "defined forms with readable depth and shadows"
        else:
            energy_desc = "explosively loud, powerful, massive, wall-of-sound, and high-energy"
            contrast_desc = "extreme high-contrast lighting, sharp chiaroscuro, blinding highlights"

        features_meta = {
            "estimated_tempo_bpm": round(tempo, 1),
            "estimated_key": detected_key,
            "spectral_centroid_hz": round(mean_centroid, 1),
            "spectral_rolloff_hz": round(mean_rolloff, 1),
            "zero_crossing_rate": round(mean_zcr, 4),
            "rms_energy_mean": round(mean_rms, 4),
            "rms_energy_std": round(std_rms, 4),
            "dynamic_range_ratio": round(dynamic_range, 4),
            "mood_type": "Minor / Introspective" if is_minor else "Major / Uplifting",
            "brightness_type": "Dark" if mean_centroid < 1200 else ("Bright" if mean_centroid > 2400 else "Balanced")
        }
        features_json = json.dumps(features_meta, indent=2)

        # Structure Prompt instructions
        prompt_style_map = {
            "Abstract Expressionism": "an abstract expressionist oil painting with wild brushstrokes, layered textures, and heavy impasto.",
            "Surrealism": "a surrealist dreamscape landscape with floating objects, melting structures, and dream-like symbolism inspired by Dali.",
            "Synthwave / Cyberpunk": "a cyberpunk digital artwork, glowing neon cyan and magenta lines, wireframe grids, and retro-futuristic city elements.",
            "Cosmic / Nebula": "a breathtaking cosmic nebula space scene, swirling interstellar dust, distant galaxies, stardust, and gas clouds.",
            "Fluid Dynamics": "a macro liquid fluid art photo, beautiful marbled acrylic swirls, organic fluid currents, and glossy polished surfaces.",
            "Orchestral Cinematic": "a cinematic fantasy landscape, epic cinematic lighting, sweeping atmospheric fog, dramatic rock formations, and hyper-detailed digital art."
        }
        
        style_template = prompt_style_map.get(art_style, "an abstract digital artwork.")

        # 1. Fallback Prompt Generator
        fallback_prompt = (
            f"Geekatplay Studio Visual Soundscape. A masterpiece depicting music in visual form: {style_template} "
            f"The sound's tempo is {tempo:.1f} BPM ({tempo_desc}), showing {movement_desc} moving across the frame. "
            f"The musical scale is {detected_key}, setting an atmosphere that is {mood_desc}. "
            f"Color palette consists of {color_desc}. "
            f"The frequency profile is {brightness_desc}, characterized by {texture_desc}. "
            f"Sonically, it is {noise_desc}, translating visually into {form_desc}. "
            f"With a volume energy that is {energy_desc}, the scene displays {contrast_desc}. "
            f"Visualized sound waves, spectrogram lines, and acoustic resonance patterns are woven into the composition. "
            f"Detailed, high-resolution, premium art, 8k, Vladimir Chopine, Geekatplay Studio style."
        )
        
        if additional_context and additional_context.strip():
            fallback_prompt += f" Incorporating elements: {additional_context.strip()}."

        # Truncate/Pad fallback prompt to be ~1000 characters if desired, but currently it is around 700-900.
        # Let's expand fallback slightly to guarantee ~1000 characters.
        if len(fallback_prompt) < 950:
            expansion = (
                f" Every brushstroke and color choice directly reflects the audio frequencies: "
                f"the low-end bass notes anchor the bottom of the scene with weight, while high-frequency treble sparkles "
                f"float like stardust at the top. The dynamic shifts are captured in the complexity of the shapes, "
                f"creating a complete synesthetic visualization where you can feel the sound vibrating through the visual medium."
            )
            fallback_prompt += expansion

        final_prompt = fallback_prompt

        # 2. Ollama Prompt Generator
        if use_ollama:
            print(f"[Geekatplay MusicMapper] Querying local Ollama model '{ollama_model}' at {ollama_url}...")
            
            system_instruction = (
                "You are an expert musicologist and prompt engineer for Geekatplay Studio by Vladimir Chopine. "
                "Your task is to write a detailed, highly descriptive prompt (about 1000 characters) for an AI image generator (like Stable Diffusion) that visually represents a piece of music based on its extracted features. "
                "Do not include any intro, outro, headers, explanation, or metadata in your response. Output ONLY the raw descriptive prompt."
            )
            
            prompt_input = (
                f"Create a detailed visual prompt based on these extracted audio features:\n"
                f"- Tempo: {tempo:.1f} BPM ({tempo_desc})\n"
                f"- Musical Key: {detected_key}\n"
                f"- Mood/Atmosphere: {mood_desc}\n"
                f"- Dominant Colors: {color_desc}\n"
                f"- Brightness/Treble Centroid: {brightness_desc} (average frequency {mean_centroid:.1f} Hz)\n"
                f"- Textures: {texture_desc}\n"
                f"- Percussiveness/Zero Crossing: {noise_desc}\n"
                f"- Energy/Volume: {energy_desc}\n"
                f"- Art Style: {style_template}\n"
                f"- User Additional Context: {additional_context if additional_context else 'None'}\n\n"
                f"Write a single descriptive paragraph (around 1000 characters) containing rich, high-fidelity art terms, detailing shapes, colors, shadows, movement, and lighting that perfectly visualize this sound. Include Geekatplay Studio and Vladimir Chopine in the description."
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
