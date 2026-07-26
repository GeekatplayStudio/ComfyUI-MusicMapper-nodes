# Geekatplay Studio: MusicMapper Nodes for ComfyUI
### Created by Vladimir Chopine

Welcome to the **Geekatplay Studio MusicMapper** node suite for ComfyUI. This extension enables users to bridge the gap between sound and vision. It provides custom nodes to transform music and audio into visually rich Mel-spectrogram images (suitable for training AI models like Riffusion or Stable Audio) and reconstruct those images back into high-quality waveforms. Additionally, it features a DSP-powered audio analyser that extracts tempo, key, brightness, and energy, feeding those metrics into local Ollama models (or a smart local fallback rules engine) to generate highly descriptive, 1000-character art prompts.

---

## 🎨 Key Features

1. **Audio to Mel-Spectrogram Conversion**:
   - Creates standard Mel-spectrogram images (grayscale or color-mapped) optimized for training AI models on music (e.g. 512x512 square format).
   - Custom **Geekatplay Orange Blue** colormap for premium, vibrant visuals.
   - Built-in duration matching (automatically crops or pads audio) so all generated spectrogram images are perfectly uniform in dimensions.
   
2. **Phase-Encoded RGB Spectrograms**:
   - Encodes magnitude in the Red channel, and complex phase details ($\cos\theta, \sin\theta$) in the Green and Blue channels.
   - Allows near-lossless reconstruction of original audio directly from the RGB image without losing phase information.

3. **Spectrogram to Audio Reconstruction**:
   - Converts spectrogram images back to raw audio waveforms.
   - Automatically detects parameters using embedded JSON metadata.
   - Employs Griffin-Lim phase reconstruction for standard Mel-spectrograms, or analytical inversion for Phase-Encoded RGB images.

4. **DSP Music Analyser & Prompt Generator**:
   - **Key/Scale Detector**: Uses a Krumhansl-Schmuckler chromagram correlation algorithm to identify the tonality (e.g. C Major, F# Minor).
   - **Tempo Estimator**: Extracts the beats-per-minute (BPM) from the audio pulse.
   - **Timbre & Energy Analysis**: Measures Spectral Centroid (brightness), Zero Crossing Rate (noise/percussiveness), and RMS Energy (dynamics).
   - **Ollama Integration**: Connects to a local Ollama server (e.g., running `llama3`, `mistral`, `gemma`, etc.) to synthesize these features into a detailed 1000-character visual prompt.
   - **Smart Fallback Engine**: If Ollama is offline, a rich, rule-based musicological template engine generates a beautiful visual prompt automatically, ensuring your workflow never breaks.

5. **Utility Nodes**:
   - **GAP Load Audio**: Loads WAV, MP3, FLAC, OGG, or M4A files into ComfyUI's standard `AUDIO` format.
   - **GAP Save Audio**: Transposes and saves generated audio waveforms directly to WAV files in the ComfyUI output directory.
   - **GAP Display Text Box**: Renders generated prompt descriptions directly inside the ComfyUI canvas.

---

## 🛠️ Installation

We provide installation scripts for both Windows PC and macOS/Linux.

### Windows (PC)
Double-click `install_deps.bat` located inside this custom node directory:
`ComfyUI/custom_nodes/ComfyUI-Geekatplay-MusicMapper/install_deps.bat`
It will automatically locate ComfyUI's python environment and install all dependencies.

### macOS / Linux
Open a terminal in this directory and execute:
```bash
chmod +x install_deps.sh
./install_deps.sh
```

### Manual Installation
If you prefer manual setup, run the following command in your ComfyUI python environment:
```bash
pip install -r requirements.txt
```

---

## 🚀 Node Reference

### 1. `GAP Load Audio`
*Loads audio files into ComfyUI.*
- **Inputs**:
  - `audio_file`: List of supported audio files in your `ComfyUI/input` folder.
- **Outputs**:
  - `AUDIO`: ComfyUI waveform tensor.
  - `STRING`: Absolute path to the file.

### 2. `GAP Save Audio`
*Saves waveforms as WAV files.*
- **Inputs**:
  - `audio`: The reconstructed audio signal.
  - `filename_prefix`: The prefix for the saved WAV file.
- **Outputs**:
  - `STRING`: Saved WAV path.

### 3. `GAP Audio To Spectrogram`
*Transforms raw audio into visual spectrogram images.*
- **Inputs**:
  - `mode`: `Mel-Spectrogram (Standard Training)` or `Phase-Encoded RGB (STFT)`.
  - `colormap`: `Geekatplay Orange Blue`, `Grayscale`, `Viridis`, `Plasma`, `Magma`, `Inferno`.
  - `n_fft`: Fast Fourier Transform window size (default: 2048).
  - `hop_length`: Step size between frames (default: 512).
  - `n_mels`: Number of Mel frequency bands (default: 512).
  - `duration`: Crop or pad audio length in seconds (default: 10.0).
  - `sample_rate`: Target sample rate (default: 44100).
  - `channel_mode`: `mixdown_mono`, `stereo_vertical`, `left_only`, `right_only`.
- **Outputs**:
  - `IMAGE`: The spectrogram image tensor `[1, H, W, 3]`.
  - `STRING`: JSON metadata containing parameters for reconstruction.

### 4. `GAP Spectrogram To Audio`
*Reconstructs audio from a spectrogram image.*
- **Inputs**:
  - `image`: Spectrogram image.
  - `reconstruct_mode`: `Auto` (reads from metadata), `Mel-Spectrogram Griffin-Lim`, or `Phase-Encoded RGB`.
  - `griffin_lim_iter`: Number of Griffin-Lim phase reconstruction iterations.
  - `metadata_json`: Optional JSON metadata (can link directly from the generator node).
- **Outputs**:
  - `AUDIO`: Reconstructed waveform.

### 5. `GAP Music Analyser & Prompt`
*Extracts audio features and generates detailed visual prompts.*
- **Inputs**:
  - `use_ollama`: Toggle local Ollama LLM integration.
  - `ollama_url`: URL of your local Ollama instance (default: `http://localhost:11434`).
  - `ollama_model`: Ollama model to use (e.g., `llama3`, `mistral`, `phi3`).
  - `art_style`: The visual style theme (e.g., `Synthwave / Cyberpunk`, `Cosmic / Nebula`, `Surrealism`).
  - `additional_context`: Additional user tags or directions.
- **Outputs**:
  - `STRING`: Detailed ~1000 character prompt.
  - `STRING`: Raw extracted features JSON string.

---

## 🦙 Ollama Configuration

To use the advanced Ollama LLM prompt generation:
1. Download and install [Ollama](https://ollama.com/).
2. Run Ollama locally on your system.
3. Download a model of your choice by running:
   ```bash
   ollama run llama3
   ```
4. In the `GAP Music Analyser & Prompt` node, check `use_ollama` as True, set the URL to `http://localhost:11434`, and set `ollama_model` to `llama3` (or the model you pulled).
5. If Ollama is offline or the model is not found, the node will seamlessly fall back to our local DSP-driven rule engine to generate the prompt.

---

## 🗺️ Included Workflows

Workflows are located in the `workflows/` folder:

### 1. Audio to Spectrogram & Music Prompt Generation (`audio_to_spectrogram_and_prompt.json`)
Loads a music file, converts it into a standard **Geekatplay Orange Blue** training spectrogram, and analyzes the audio. It generates a detailed description using Ollama (or fallback) and displays it on screen. It then passes the generated prompt into a standard KSampler to generate an image representing the music.

### 2. Spectrogram Image to Audio Reconstruction (`spectrogram_to_audio.json`)
Loads a saved spectrogram image (containing embedded or linked metadata), processes it through the Griffin-Lim or Phase-Reconstruction decoder, and exports the reconstructed sound back into a playable WAV file.

---

## 🏷️ Credits & Branding
*Designed and branded for **Geekatplay Studio by Vladimir Chopine**.*
*Learn more at [Geekatplay Studio](https://www.geekatplay.com).*
