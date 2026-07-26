# Geekatplay Studio: MusicMapper Nodes for ComfyUI
### Created by Vladimir Chopine
*Lean, specialized custom nodes for converting music into AI-trainable Mel-spectrogram images, analyzing audio features into 1000-character musicological reports, and reconstructing audio from spectrograms. Seamlessly integrates with native ComfyUI audio and image nodes.*

---

## 🎵 Overview

**Geekatplay Studio MusicMapper** is built to complement native ComfyUI functionality. Instead of reinventing standard file loading or saving utilities, this package focuses on **3 powerful, specialized custom nodes**:

1. **`GAP Audio To Spectrogram`**: Converts native audio into square 512x512 Mel-spectrogram images (grayscale or color-mapped with our signature **Geekatplay Orange Blue** theme) ready for model training, or Phase-Encoded RGB images for lossless audio reconstruction.
2. **`GAP Spectrogram To Audio`**: Reconstructs audio from Mel-spectrogram images (via Griffin-Lim) or Phase-Encoded RGB images back into native ComfyUI audio waveforms.
3. **`GAP Music Analyser & Prompt`**: Extracts musical & acoustic metrics (Key/Scale, BPM, Spectral Centroid brightness, Zero Crossing Rate, RMS dynamic energy) and leverages local Ollama LLMs (or our built-in offline rules engine) to synthesize an extensive ~1000-character pure musicological description.

---

## 📖 Complete Beginner's Guide

### Step 1: Install Git

#### 🪟 Windows (PC):
Download Git from [https://git-scm.com/download/win](https://git-scm.com/download/win) or run:
```cmd
winget install --id Git.Git -e --source winget
```

#### 🍎 macOS (Mac):
Open Terminal (`Cmd + Space` -> `Terminal`) and run:
```bash
xcode-select --install
```

---

### Step 2: Install Ollama & Download AI Models (Optional for Prompt Generation)

Ollama runs local LLMs for music analysis synthesis.

1. Download Ollama from [https://ollama.com](https://ollama.com).
2. Open Command Prompt (PC) or Terminal (Mac) and pull a model:
   ```bash
   ollama pull llama3
   ```
*(Note: If Ollama is not installed, `GAP Music Analyser` automatically uses its built-in offline DSP rules engine so your workflows run seamlessly.)*

---

### Step 3: Clone the Custom Nodes

Open Terminal / Command Prompt, navigate to your `ComfyUI/custom_nodes` folder, and clone the repo:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GeekatplayStudio/ComfyUI-MusicMapper-nodes.git
```

---

### Step 4: Install Python Dependencies

Run the automated installer script inside the node folder:
- **Windows**: Double-click `install_deps.bat`
- **macOS / Linux**: Run `chmod +x install_deps.sh && ./install_deps.sh`

---

### Step 5: Start ComfyUI & Drag-and-Drop Workflows

Start ComfyUI and drag-and-drop the JSON files located in the `workflows/` directory:

#### 1. Audio to Spectrogram & Music Analysis (`workflows/audio_to_spectrogram_and_prompt.json`)
- Uses native ComfyUI **`LoadAudio`** node to pick/upload music.
- Passes audio into **`GAP Music Analyser`** for ~1000-char music analysis.
- Passes audio into **`GAP Audio To Spectrogram`** to render a 512x512 training image.
- Uses native ComfyUI **`SaveImage`** to export the spectrogram.

#### 2. Spectrogram Image to Audio Reconstruction (`workflows/spectrogram_to_audio.json`)
- Uses native ComfyUI **`LoadImage`** node to pick a spectrogram PNG.
- Passes the image into **`GAP Spectrogram To Audio`** to synthesize the audio waveform.
- Uses native ComfyUI **`SaveAudio`** to preview and export the WAV file.

---

## 🧩 Custom Nodes Reference

| Node Name | Category | Function |
| :--- | :--- | :--- |
| **`GAP Audio To Spectrogram`** | `Geekatplay Studio/Spectrogram` | Converts audio into Mel-spectrogram images (512x512) or Phase-Encoded RGB images. |
| **`GAP Spectrogram To Audio`** | `Geekatplay Studio/Spectrogram` | Reconstructs audio from spectrogram images using Griffin-Lim or Phase Inversion. |
| **`GAP Music Analyser & Prompt`** | `Geekatplay Studio/Audio` | Extracts Key, BPM, Centroid, and RMS metrics and synthesizes a ~1000-char musicological report via Ollama or offline DSP. |

---

## 🏷️ Credits & License

Designed and developed for **Geekatplay Studio by Vladimir Chopine**.  
Visit [Geekatplay Studio](https://www.geekatplay.com) for digital art resources and tutorials.
