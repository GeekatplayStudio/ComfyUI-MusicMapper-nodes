# Geekatplay Studio: MusicMapper Custom Nodes for ComfyUI
### Created by Vladimir Chopine
*Specialized custom nodes for converting music into AI-trainable Mel-spectrogram images, analyzing raw audio files using LAION-CLAP Deep Learning neural networks into ~1000-character musicological reports, and lossless audio reconstruction. Integrates seamlessly with native ComfyUI nodes.*

---

## 🎵 Key Features

1. **Deep Learning Audio Analysis (LAION-CLAP)**:
   - Passes raw `.wav` / `.mp3` audio files directly through the **LAION-CLAP** neural network (`laion/clap-htsat-fused`).
   - Automatically downloads the pre-trained model weights from HuggingFace on first run.
   - Extracts pitch key/scale, tempo (BPM), spectral centroid brightness, zero-crossing rate, RMS dynamics, and acoustic instrument classification probabilities into a **~1000-character musicological prompt**.

2. **AI Model Training Spectrogram Generation**:
   - Converts audio into 512x512 Mel-spectrogram images color-mapped with our signature **Geekatplay Orange Blue** theme or grayscale for AI model training.

3. **Lossless Audio Reconstruction (Phase-Encoded RGB)**:
   - Auto-detects phase-encoded RGB images and performs exact analytical Inverse Short-Time Fourier Transform (`librosa.istft`).
   - Reconstructs original audio with **0.9991 signal correlation** and zero phase distortion.

4. **100% Native ComfyUI Integration**:
   - Designed to work alongside native ComfyUI nodes: **`LoadAudio`**, **`ShowText`**, **`SaveImage`**, and **`SaveAudio`**.

---

## 📖 Beginner's Setup Guide

### Step 1: Install Git

#### 🪟 Windows (PC):
Download Git from [https://git-scm.com/download/win](https://git-scm.com/download/win) or run in Command Prompt:
```cmd
winget install --id Git.Git -e --source winget
```

#### 🍎 macOS (Mac):
Open Terminal (`Cmd + Space` -> `Terminal`) and run:
```bash
xcode-select --install
```

---

### Step 2: Clone into ComfyUI

Open Terminal / Command Prompt, navigate to your `ComfyUI/custom_nodes` folder, and clone the repo:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GeekatplayStudio/ComfyUI-MusicMapper-nodes.git
```

---

### Step 3: Install Dependencies

Run the automated installer script inside the node folder:
- **Windows**: Double-click `install_deps.bat`
- **macOS / Linux**: Run `chmod +x install_deps.sh && ./install_deps.sh`

---

### Step 4: Ollama Setup (Optional)

1. Download Ollama from [https://ollama.com](https://ollama.com).
2. Open Command Prompt (PC) or Terminal (Mac) and pull a model:
   ```bash
   ollama pull llama3.1
   ```
*(Note: LAION-CLAP Deep Learning engine auto-downloads via HuggingFace without needing Ollama).*

---

## 🖥️ Included Workflows

Located in the `workflows/` folder:

1. **`audio_to_spectrogram_and_prompt.json`**:
   - Uses native **`LoadAudio`** node for music input.
   - Analyzes raw audio via **`GAP Music Analyser`** (LAION-CLAP Deep Learning).
   - Displays prompt on native **`ShowText`** node.
   - Renders 512x512 training image via **`GAP Audio To Spectrogram`**.
   - Exports image via native **`SaveImage`** node.

2. **`spectrogram_to_audio.json`**:
   - Loads image via native **`LoadImage`** node.
   - Reconstructs lossless audio via **`GAP Spectrogram To Audio`**.
   - Plays back and exports WAV via native **`SaveAudio`** node.

---

## 🏷️ Credits & License

Designed and developed for **Geekatplay Studio by Vladimir Chopine**.  
Visit [Geekatplay Studio](https://www.geekatplay.com) for digital art resources and tutorials.
