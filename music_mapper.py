import os
import json
import numpy as np
import torch
import librosa
import soundfile as sf
import requests
import matplotlib.pyplot as plt
import folder_paths

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Setup default configurations
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 512
EPSILON = 1e-8

def estimate_key(y, sr):
    """
    Estimates the musical key and scale of the audio using Krumhansl-Schmuckler profile correlation.
    """
    try:
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
        minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)

        best_corr = -999.0
        best_key = "C Major"

        for shift in range(12):
            shifted_chroma = np.roll(chroma_mean, -shift)
            
            if np.std(shifted_chroma) > 0:
                shifted_chroma = (shifted_chroma - np.mean(shifted_chroma)) / np.std(shifted_chroma)
            else:
                shifted_chroma = shifted_chroma - np.mean(shifted_chroma)

            corr_maj = np.dot(shifted_chroma, major_profile)
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key = f"{notes[shift]} Major"

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
            return rgba[..., :3]
        except Exception as e:
            print(f"[Geekatplay MusicMapper] Error applying colormap '{colormap_name}': {e}. Falling back to Grayscale.")
            return np.stack([data_norm, data_norm, data_norm], axis=-1)

def invert_colormap_to_grayscale(rgb_img):
    """
    Converts RGB image back to grayscale using luminance standard (ITU-R BT.601).
    """
    r, g, b = rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


class GeekatplayLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'))]
        if not files:
            files = ["example.wav"]
        return {
            "required": {
                "audio_file": (sorted(files), {"audio_upload": True}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "load_audio"
    CATEGORY = "Geekatplay Studio/Audio"

    def load_audio(self, audio_file, custom_path=""):
        resolved_path = ""
        if custom_path and custom_path.strip():
            cp = custom_path.strip().strip('"').strip("'")
            if os.path.isfile(cp):
                resolved_path = cp
            else:
                input_dir = folder_paths.get_input_directory()
                joined = os.path.join(input_dir, cp)
                if os.path.isfile(joined):
                    resolved_path = joined

        if not resolved_path:
            input_dir = folder_paths.get_input_directory()
            annotated = folder_paths.get_annotated_filepath(audio_file)
            if os.path.isfile(annotated):
                resolved_path = annotated
            else:
                joined = os.path.join(input_dir, audio_file)
                if os.path.isfile(joined):
                    resolved_path = joined

        if not resolved_path or not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Audio file not found: '{audio_file}' / '{custom_path}'")

        y, sr = librosa.load(resolved_path, sr=None, mono=False)
        if y.ndim == 1:
            waveform = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
        else:
            waveform = torch.from_numpy(y).unsqueeze(0)

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
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        y_tensor = waveform[0].cpu()
        y_np = y_tensor.numpy().T
        
        max_val = np.max(np.abs(y_np))
        if max_val > 1.0:
            y_np = y_np / max_val
        elif max_val > 0.001 and max_val < 0.3:
            y_np = (y_np / max_val) * 0.95
            
        y_np = np.nan_to_num(y_np, nan=0.0)
        
        output_dir = folder_paths.get_output_directory()
        import uuid
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.wav"
        save_path = os.path.join(output_dir, filename)
        
        sf.write(save_path, y_np, sample_rate)
        results = [{"filename": filename, "subfolder": "", "type": "output"}]
        return {"ui": {"audio": results}, "result": (save_path,)}


class GeekatplayPreviewAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("temp_path",)
    FUNCTION = "preview_audio"
    CATEGORY = "Geekatplay Studio/Audio"
    OUTPUT_NODE = True

    def preview_audio(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        y_tensor = waveform[0].cpu()
        y_np = y_tensor.numpy().T
        
        max_val = np.max(np.abs(y_np))
        if max_val > 1.0:
            y_np = y_np / max_val
        elif max_val > 0.001 and max_val < 0.3:
            y_np = (y_np / max_val) * 0.95
            
        y_np = np.nan_to_num(y_np, nan=0.0)
        
        temp_dir = folder_paths.get_temp_directory()
        import uuid
        filename = f"GAP_Preview_{uuid.uuid4().hex[:8]}.wav"
        save_path = os.path.join(temp_dir, filename)
        
        sf.write(save_path, y_np, sample_rate)
        results = [{"filename": filename, "subfolder": "", "type": "temp"}]
        return {"ui": {"audio": results}, "result": (save_path,)}


class GeekatplayDisplayTextBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "display_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_text"
    CATEGORY = "Geekatplay Studio/Audio"
    OUTPUT_NODE = True

    def display_text(self, text, display_text=""):
        print("\n==========================================================")
        print("[Geekatplay MusicMapper] Display Text Box Output:")
        print(text)
        print("==========================================================\n")
        return {"ui": {"text": [text], "string": [text]}, "result": (text,)}


class GeekatplayAudioToSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["Phase-Encoded RGB (STFT)", "Mel-Spectrogram (Standard Training)"], {"default": "Phase-Encoded RGB (STFT)"}),
                "colormap": (["Geekatplay Orange Blue", "Grayscale", "Viridis", "Plasma", "Magma", "Inferno"], {"default": "Geekatplay Orange Blue"}),
                "n_fft": ("INT", {"default": DEFAULT_N_FFT, "min": 256, "max": 8192, "step": 256}),
                "hop_length": ("INT", {"default": DEFAULT_HOP_LENGTH, "min": 64, "max": 4096, "step": 64}),
                "n_mels": ("INT", {"default": DEFAULT_N_MELS, "min": 64, "max": 1024, "step": 64}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 6000.0, "step": 1.0}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000}),
                "channel_mode": (["mixdown_mono", "stereo_vertical", "left_only", "right_only"], {"default": "mixdown_mono"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "generate_spectrogram"
    CATEGORY = "Geekatplay Studio/Spectrogram"

    def generate_spectrogram(self, audio, mode, colormap, n_fft, hop_length, n_mels, duration, sample_rate, channel_mode):
        waveform = audio["waveform"][0].cpu().numpy()
        orig_sr = audio["sample_rate"]
        
        if orig_sr != sample_rate:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sample_rate)
            
        if channel_mode == "mixdown_mono":
            y = np.mean(waveform, axis=0, keepdims=True)
        elif channel_mode == "left_only":
            y = waveform[0:1]
        elif channel_mode == "right_only":
            y = waveform[1:2] if waveform.shape[0] > 1 else waveform[0:1]
        else:
            y = waveform[0:2]

        num_channels = y.shape[0]
        if duration > 0.0:
            target_samples = int(duration * sample_rate)
            if y.shape[-1] < target_samples:
                pad_width = target_samples - y.shape[-1]
                y = np.pad(y, ((0, 0), (0, pad_width)), mode='constant')
            elif y.shape[-1] > target_samples:
                y = y[:, :target_samples]

        panels = []
        metadata_channels = []
        
        for c in range(num_channels):
            channel_y = y[c]
            
            if mode == "Mel-Spectrogram (Standard Training)":
                S = librosa.feature.melspectrogram(
                    y=channel_y, 
                    sr=sample_rate, 
                    n_fft=n_fft, 
                    hop_length=hop_length, 
                    n_mels=n_mels,
                    fmax=sample_rate / 2.0
                )
                S_db = librosa.power_to_db(S, ref=1.0)
                
                db_min = float(S_db.min())
                db_max = float(S_db.max())
                db_span = db_max - db_min
                if db_span < EPSILON:
                    db_span = EPSILON
                
                S_norm = (S_db - db_min) / db_span
                panel = apply_colormap(S_norm, colormap)
                panel = np.flipud(panel)
                panels.append(panel)
                
                metadata_channels.append({
                    "channel_index": c,
                    "db_min": db_min,
                    "db_max": db_max
                })
                
            else: # Phase-Encoded RGB (STFT)
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
                phase_cos = (np.cos(phase) + 1.0) * 0.5
                phase_sin = (np.sin(phase) + 1.0) * 0.5
                
                panel = np.stack([mag_norm, phase_cos, phase_sin], axis=-1)
                panel = np.flipud(panel)
                panels.append(panel)
                
                metadata_channels.append({
                    "channel_index": c,
                    "mag_min": mag_min,
                    "mag_max": mag_max
                })

        final_img = np.vstack(panels)
        image_tensor = torch.from_numpy(final_img).unsqueeze(0).float()

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
        meta = {}
        if metadata_json and metadata_json.strip():
            try:
                meta = json.loads(metadata_json)
            except Exception as e:
                print(f"[Geekatplay MusicMapper] Error parsing metadata: {e}")

        # Resolve mode: Auto-detect Phase-Encoded RGB if Green/Blue channels contain phase variance
        img_np = image[0].cpu().numpy() # [H, W, C]
        
        mode = meta.get("mode", "")
        if not mode or reconstruct_mode != "Auto":
            if reconstruct_mode == "Mel-Spectrogram Griffin-Lim":
                mode = "Mel-Spectrogram (Standard Training)"
            elif reconstruct_mode == "Phase-Encoded RGB":
                mode = "Phase-Encoded RGB (STFT)"
            else:
                # Auto-detect from image channels: if RGB has phase variance in G/B channels, use Phase-Encoded RGB!
                if img_np.ndim == 3 and img_np.shape[-1] >= 3:
                    g_var = float(np.var(img_np[..., 1]))
                    b_var = float(np.var(img_np[..., 2]))
                    if g_var > 0.002 and b_var > 0.002:
                        mode = "Phase-Encoded RGB (STFT)"
                        print("[Geekatplay MusicMapper] Auto-detected Phase-Encoded RGB image! Using Lossless ISTFT Reconstruction.")
                    else:
                        mode = "Mel-Spectrogram (Standard Training)"
                else:
                    mode = "Mel-Spectrogram (Standard Training)"

        sr = meta.get("sample_rate", sample_rate)
        fft_len = meta.get("n_fft", n_fft)
        hop = meta.get("hop_length", hop_length)
        channels = meta.get("channels", 1)
        original_samples = meta.get("original_samples", 0)
        channel_metadata = meta.get("channel_metadata", [])

        panel_height = img_np.shape[0] // channels
        reconstructed_waveforms = []
        
        for c in range(channels):
            start_y = c * panel_height
            end_y = start_y + panel_height
            panel = img_np[start_y:end_y]
            panel = np.flipud(panel)
            
            c_meta = channel_metadata[c] if c < len(channel_metadata) else {}
            
            if mode == "Mel-Spectrogram (Standard Training)":
                db_min = c_meta.get("db_min", -80.0)
                db_max = c_meta.get("db_max", 0.0)
                
                S_norm = invert_colormap_to_grayscale(panel)
                S_db = S_norm * (db_max - db_min) + db_min
                S = librosa.db_to_power(S_db)
                
                y_recon = librosa.feature.inverse.mel_to_audio(
                    S,
                    sr=sr,
                    n_fft=fft_len,
                    hop_length=hop,
                    n_iter=max(griffin_lim_iter, 64)
                )
                
            else: # Phase-Encoded RGB (Lossless STFT)
                mag_min = c_meta.get("mag_min", 0.0)
                mag_max = c_meta.get("mag_max", 10.0)
                
                mag_norm = panel[..., 0]
                log_magnitude = mag_norm * (mag_max - mag_min) + mag_min
                magnitude = np.expm1(log_magnitude)
                
                phase_cos = panel[..., 1] * 2.0 - 1.0
                phase_sin = panel[..., 2] * 2.0 - 1.0
                phase = np.arctan2(phase_sin, phase_cos)
                
                D = magnitude * np.exp(1j * phase)
                n_fft_actual = 2 * (D.shape[0] - 1)
                y_recon = librosa.istft(
                    D,
                    n_fft=n_fft_actual,
                    hop_length=hop,
                    win_length=n_fft_actual
                )

            if original_samples > 0:
                if len(y_recon) > original_samples:
                    y_recon = y_recon[:original_samples]
                elif len(y_recon) < original_samples:
                    y_recon = np.pad(y_recon, (0, original_samples - len(y_recon)), mode='constant')
                
            reconstructed_waveforms.append(y_recon)
            
        waveform_np = np.stack(reconstructed_waveforms, axis=0)
        
        # Only normalize if peak amplitude exceeds 1.0 to preserve exact dynamics
        max_val = np.max(np.abs(waveform_np))
        if max_val > 1.0:
            waveform_np = waveform_np / max_val
        elif max_val < 0.01 and mode == "Mel-Spectrogram (Standard Training)":
            waveform_np = (waveform_np / max_val) * 0.95
            
        waveform_np = np.nan_to_num(waveform_np, nan=0.0)
        
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
                "analysis_engine": (["LAION-CLAP Deep Learning (Auto-Download)", "Ollama LLM", "Offline DSP Rules"], {"default": "LAION-CLAP Deep Learning (Auto-Download)"}),
                "prompt_style": (["Suno / Udio Style (Comma-Separated Tags)", "Detailed Musicological Report"], {"default": "Suno / Udio Style (Comma-Separated Tags)"}),
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
    OUTPUT_NODE = True

    def analyze_music(self, audio, analysis_engine="LAION-CLAP Deep Learning (Auto-Download)", prompt_style="Suno / Udio Style (Comma-Separated Tags)", ollama_url="http://localhost:11434", ollama_model="llama3", additional_context=""):
        waveform = audio["waveform"][0].cpu().numpy()
        sr = audio["sample_rate"]
        y = np.mean(waveform, axis=0)
        
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
            
        print("[Geekatplay MusicMapper] Estimating Key and Scale...")
        detected_key = estimate_key(y, sr)
        
        print("[Geekatplay MusicMapper] Performing Spectral Feature Extraction...")
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(centroid))
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))
        
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        dynamic_range = std_rms / (mean_rms + EPSILON)
        
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

        is_minor = "minor" in detected_key.lower()
        if is_minor:
            tonality_desc = "introspective, emotionally somber, minor-key harmonic structure"
            consonance_desc = "complex harmonic tension, dark modal undertones, and brooding chord progressions"
        else:
            tonality_desc = "bright, uplifting, major-key harmonic structure"
            consonance_desc = "resonant consonant harmonies, stable tonal centers, and triumphant chord voicings"

        if mean_centroid < 1200:
            brightness_desc = "deep, sub-bass heavy, warm timbral profile with low spectral centroid"
            freq_balance = "heavy fundamental bass energy, muted upper harmonics, and dark warm timbre"
        elif mean_centroid < 2400:
            brightness_desc = "balanced mid-range timbral profile with natural spectral distribution"
            freq_balance = "well-defined vocal and instrumental mid-range frequencies with organic acoustic balance"
        else:
            brightness_desc = "sparkling, bright, treble-dominant timbral profile with high spectral centroid"
            freq_balance = "crisp high-frequency overtones, sharp metallic sibilance, and brilliant presence"

        if mean_zcr < 0.04:
            timbre_desc = "purely tonal, smooth sinusoidal harmonics, and clean melodic contours"
            percussion_desc = "minimal transient noise, dominated by legato melodic lines and pure acoustic tones"
        elif mean_zcr < 0.12:
            timbre_desc = "balanced acoustic texture with mixed tonal and percussive transients"
            percussion_desc = "defined drum hits, articulate articulation, and balanced harmonic noise"
        else:
            timbre_desc = "highly percussive, noisy, distorted, or heavily transient acoustic profile"
            percussion_desc = "sharp attack transients, dense noise bursts, complex percussive articulation, and static grain"

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
            fallback_prompt += (
                f" This detailed audio feature signature accurately describes the physical sound waves, tonal key center, "
                f"frequency distribution, transient density, and dynamic loudness of the recording for model mapping and acoustic synthesis."
            )

        final_prompt = fallback_prompt

        if analysis_engine == "LAION-CLAP Deep Learning (Auto-Download)":
            print("[Geekatplay MusicMapper] Running LAION-CLAP Deep Learning Audio AI Model...")
            try:
                from transformers import ClapModel, ClapProcessor  # type: ignore # noqa: F401
                
                if sr != 48000:
                    y_48k = librosa.resample(y, orig_sr=sr, target_sr=48000)
                else:
                    y_48k = y
                    
                print("[Geekatplay MusicMapper] Loading HuggingFace LAION-CLAP model (laion/clap-htsat-fused)...")
                processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
                model = ClapModel.from_pretrained("laion/clap-htsat-fused")
                
                candidate_labels = [
                    "80s synthpop darkwave electronic rock",
                    "heavy distorted electric guitar riff rock",
                    "acoustic piano ballad performance",
                    "analog synthesizer bassline synthwave",
                    "driving electronic drum rhythm beat",
                    "symphonic cinematic orchestral strings",
                    "ambient atmospheric pad soundscape",
                    "funky bass groove pop music",
                    "hard rock heavy metal guitar"
                ]
                
                inputs = processor(audio=y_48k, text=candidate_labels, sampling_rate=48000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_audio = outputs.logits_per_audio
                    probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
                    
                top_idx = int(np.argmax(probs))
                top_label = candidate_labels[top_idx]
                top_prob = float(probs[top_idx])
                
                print(f"[Geekatplay MusicMapper] CLAP Primary Acoustic Classification: '{top_label}' ({top_prob*100:.1f}% confidence)")
                
                features_meta["clap_primary_classification"] = top_label
                features_meta["clap_classification_confidence"] = round(top_prob, 4)
                
                clap_report = (
                    f"Geekatplay Studio Deep Learning Audio Analysis Report by Vladimir Chopine. "
                    f"Multimodal Audio Model Feature Analysis: The raw audio recording was processed directly through the LAION-CLAP deep neural network architecture. "
                    f"The neural model identifies the primary acoustic sound signature as '{top_label}' with {top_prob*100:.1f}% classification confidence. "
                    f"Harmonically, the track centers around the key of {detected_key} ({tonality_desc}), exhibiting {consonance_desc}. "
                    f"Rhythmically, the audio signal tracks at an estimated tempo of {tempo:.1f} BPM ({tempo_desc}), displaying {rhythm_pattern}. "
                    f"Spectrally, the audio centroid averages {mean_centroid:.1f} Hz ({brightness_desc}), demonstrating {freq_balance}. "
                    f"Timbrally, the signal registers a zero crossing rate of {mean_zcr:.4f} ({timbre_desc}), featuring {percussion_desc}. "
                    f"Dynamically, with a mean RMS energy level of {mean_rms:.4f} ({dynamic_desc}), the amplitude envelope exhibits {amplitude_envelope}. "
                    f"Spectral rolloff cutoff is at {mean_rolloff:.1f} Hz with a dynamic range ratio of {dynamic_range:.4f}."
                )
                if additional_context and additional_context.strip():
                    clap_report += f" Additional Context: {additional_context.strip()}."
                    
                if len(clap_report) < 950:
                    clap_report += (
                        " This deep learning audio feature embedding signature captures the physical acoustic sound waves, "
                        "instrumental timbres, pitch key center, and dynamic energy density of the recording for model training and sound mapping."
                    )
                final_prompt = clap_report
            except Exception as e:
                print(f"[Geekatplay MusicMapper] LAION-CLAP execution error: {e}. Falling back to rule-based engine.")

        elif analysis_engine == "Ollama LLM":
            print(f"[Geekatplay MusicMapper] Querying local Ollama model '{ollama_model}' at {ollama_url}...")
            system_instruction = (
                "You are a professional music prompt engineer for Suno AI and Udio by Vladimir Chopine at Geekatplay Studio. "
                "Your task is to write a rich, highly detailed, comma-separated list of musical style tags, genres, instrumentation, tempo, key, mood, and production techniques based on the provided DSP audio features. "
                "Do NOT include any intro text, conversational filler, or formatting headers. Output ONLY the raw comma-separated prompt tags."
            )
            prompt_input = (
                f"Generate a rich comma-separated prompt for Suno / Udio based on these extracted DSP audio features:\n"
                f"- Tempo: {tempo:.1f} BPM\n"
                f"- Key & Scale: {detected_key}\n"
                f"- Primary Genre / Sound Signature: {features_meta.get('clap_primary_classification', 'electronic rock')}\n"
                f"- Spectral Centroid Brightness: {mean_centroid:.1f} Hz ({brightness_desc})\n"
                f"- Dynamic Energy Level (RMS): {mean_rms:.4f} ({dynamic_desc})\n"
                f"- Additional Context: {additional_context if additional_context else 'None'}\n"
            )
            try:
                url = f"{ollama_url.rstrip('/')}/api/generate"
                payload = {
                    "model": ollama_model,
                    "prompt": f"{system_instruction}\n\nUser Request:\n{prompt_input}",
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 180}
                }
                response = requests.post(url, json=payload, timeout=6.0)
                if response.status_code == 200:
                    resp_json = response.json()
                    response_text = resp_json.get("response", "")
                    if not response_text and "message" in resp_json:
                        response_text = resp_json["message"].get("content", "")
                    response_text = response_text.strip()
                    if len(response_text) > 10:
                        final_prompt = response_text
                        print("[Geekatplay MusicMapper] Ollama prompt generated successfully.")
            except Exception as e:
                print(f"[Geekatplay MusicMapper] Ollama error: {e}. Using rule-based fallback.")

        if prompt_style == "Suno / Udio Style (Comma-Separated Tags)":
            suno_tags = []
            
            # 1. Genre & Primary Acoustic Sound Signature
            primary = features_meta.get("clap_primary_classification", "")
            if primary:
                suno_tags.append(primary)
            else:
                if mean_centroid > 3000:
                    suno_tags.append("80s synthpop, darkwave, electronic rock")
                elif mean_centroid < 1200:
                    suno_tags.append("deep bass synthwave, low-end groove")
                else:
                    suno_tags.append("melodic acoustic rock performance")

            # 2. Key & Tempo
            suno_tags.append(f"{tempo:.1f} BPM")
            suno_tags.append(f"key of {detected_key}")
            
            # 3. Harmonic & Emotional Tonality
            if is_minor:
                suno_tags.append("introspective, emotional minor key, dark modal chord progression")
            else:
                suno_tags.append("bright, triumphant major-key progression, uplifting harmony")

            # 4. Timbral Profile & High-Frequency Details
            if mean_centroid > 3000:
                suno_tags.append("sparkling bright treble, crisp metallic overtones, sharp high-frequency presence")
            elif mean_centroid < 1200:
                suno_tags.append("deep sub-bass warm timbre, heavy fundamental bass energy")
            else:
                suno_tags.append("balanced organic mid-range balance, natural frequency distribution")

            # 5. Rhythm & Transient Texture
            if mean_zcr > 0.06:
                suno_tags.append("driving percussive attack, sharp transients, articulate drum rhythm")
            else:
                suno_tags.append("smooth legato melodic lines, clean sinusoidal harmonics")

            # 6. Dynamic Range & Wall-of-Sound Production
            if mean_rms > 0.08:
                suno_tags.append("high-energy wall-of-sound compression, loud dynamic punch, full-scale mastering")
            elif mean_rms < 0.015:
                suno_tags.append("whisper-soft intimate dynamics, delicate low-amplitude texture")
            else:
                suno_tags.append("steady dynamic headroom, balanced sound pressure level")

            # 7. Production Aesthetics & Additional Context
            suno_tags.append("stereo width, professional studio mix")
            if additional_context and additional_context.strip():
                suno_tags.append(additional_context.strip())

            final_prompt = ", ".join(suno_tags)

        features_json = json.dumps(features_meta, indent=2)
        return {
            "ui": {
                "text": [final_prompt],
                "string": [final_prompt]
            },
            "result": (final_prompt, features_json)
        }
