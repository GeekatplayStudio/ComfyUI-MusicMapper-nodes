# Geekatplay Studio: MusicMapper Nodes for ComfyUI
### Created by Vladimir Chopine
*The ultimate custom node suite for converting music and sound into AI-trainable Mel-spectrogram images, analyzing audio features, and generating detailed ~1000-character visual prompts via local Ollama LLMs.*

---

## 🎵 Overview

**Geekatplay Studio MusicMapper** connects the worlds of audio and visual AI generation in ComfyUI. 

Whether you are training AI music generation models (like Riffusion or Stable Audio) or creating visual artwork inspired by music, this node suite provides:
1. **Audio-to-Spectrogram Conversion**: Generates square 512x512 Mel-spectrogram images (grayscale or color-mapped with our signature **Geekatplay Orange Blue** theme) ready for model training.
2. **Phase-Encoded RGB Spectrograms**: Encodes complex sound phase data into RGB channels for near-lossless audio reconstruction.
3. **Spectrogram-to-Audio Reconstruction**: Converts spectrogram images back into playable WAV audio files using Griffin-Lim or analytical phase inversion.
4. **DSP Music Analyser & Prompt Generator**: Extracts key/scale, tempo (BPM), spectral centroid (brightness), zero crossing rate (noisiness), and RMS energy (volume).
5. **Local Ollama LLM Integration**: Feeds extracted audio metrics into local Ollama models (e.g. `llama3`, `mistral`, `phi3`) to generate rich ~1000-character visual prompts for image generation. Includes a local offline fallback engine if Ollama is not active.

---

## 📖 Complete Beginner's Installation Guide

If you have never installed custom nodes, Git, or Ollama before, follow this guide step-by-step.

### Step 1: Install Git (Required to download custom nodes)

Git is a free tool used to download code repositories from GitHub.

#### 🪟 On Windows (PC):
1. Download **Git for Windows** from the official site: [https://git-scm.com/download/win](https://git-scm.com/download/win).
2. Run the downloaded `.exe` installer. Click **Next** on all prompts to accept default settings and click **Install**.
3. *Alternative via Command Prompt (winget)*: Open Command Prompt as Administrator and run:
   ```cmd
   winget install --id Git.Git -e --source winget
   ```

#### 🍎 On macOS (Mac):
1. Open the **Terminal** app (Press `Cmd + Space`, type `Terminal`, and press `Enter`).
2. Run the following command to install the Apple Developer Command Line Tools (which includes Git):
   ```bash
   xcode-select --install
   ```
3. A popup window will appear. Click **Install** and accept the license terms.
4. *Alternative via Homebrew*: If you use Homebrew, run:
   ```bash
   brew install git
   ```

---

### Step 2: Install Ollama & Download AI Models (For Music Prompts)

Ollama runs open-source LLMs locally on your computer (no cloud subscription needed).

#### 🪟 On Windows (PC):
1. Go to [https://ollama.com/download/windows](https://ollama.com/download/windows).
2. Download and run `OllamaSetup.exe`.
3. Once installed, open your **Command Prompt** or **PowerShell** and pull a recommended LLM model:
   ```cmd
   ollama pull llama3
   ```
   *(Optional additional models: `ollama pull mistral` or `ollama pull phi3`)*
4. Verify Ollama is running by opening your browser and visiting `http://localhost:11434`. You should see the message: `"Ollama is running"`.

#### 🍎 On macOS (Mac):
1. Go to [https://ollama.com/download/mac](https://ollama.com/download/mac).
2. Download the `.zip` file, unzip it, and drag the **Ollama** app into your `Applications` folder.
3. Open **Ollama** from your Applications folder.
4. Open **Terminal** and run:
   ```bash
   ollama pull llama3
   ```
5. Verify it is running by checking `http://localhost:11434` in Safari or Chrome.

> [!NOTE]
> **Is Ollama strictly required?** No! If Ollama is not installed or turned off, our `GAP Music Analyser` node automatically uses its built-in **DSP Rules Engine** to generate the ~1000-character prompt offline. You will never get an error!

---

### Step 3: Clone the MusicMapper Custom Nodes

1. Open **Command Prompt** (Windows) or **Terminal** (Mac).
2. Navigate to your ComfyUI `custom_nodes` folder:

   - **On Windows**:
     ```cmd
     cd /d C:\path\to\your\ComfyUI\custom_nodes
     ```
     *(Replace `C:\path\to\your\ComfyUI` with your actual ComfyUI directory path)*

   - **On macOS**:
     ```bash
     cd /path/to/your/ComfyUI/custom_nodes
     ```

3. Run the `git clone` command to download the repository:
   ```bash
   git clone https://github.com/GeekatplayStudio/ComfyUI-MusicMapper-nodes.git
   ```

---

### Step 4: Install Python Dependencies

Our nodes use specialized audio processing libraries (`librosa`, `soundfile`, `matplotlib`).

#### 🪟 On Windows (PC):
1. Open File Explorer and navigate to your `ComfyUI/custom_nodes/ComfyUI-Geekatplay-MusicMapper/` folder.
2. Double-click the file named **`install_deps.bat`**.
3. A command window will open, automatically detect your ComfyUI python environment, install `requirements.txt`, and display: `"Installation complete successfully!"`.

#### 🍎 On macOS (Mac):
1. Open **Terminal** and navigate into the node folder:
   ```bash
   cd /path/to/your/ComfyUI/custom_nodes/ComfyUI-Geekatplay-MusicMapper
   ```
2. Make the installer script executable and run it:
   ```bash
   chmod +x install_deps.sh
   ./install_deps.sh
   ```

---

### Step 5: Start ComfyUI & Load Workflows

1. Start (or restart) your ComfyUI server.
2. Inside ComfyUI, you will see a new node category: **`Geekatplay Studio`**.
3. We provide two ready-to-use visual workflow JSON files in the `workflows/` directory:

#### Workflow 1: Audio to Spectrogram & Music Prompt Generation
- **File**: `workflows/audio_to_spectrogram_and_prompt.json`
- **What it does**:
  1. Loads your input audio file (WAV, MP3, FLAC).
  2. Analyzes key, scale, tempo, brightness, and volume energy.
  3. Queries local Ollama to generate a detailed ~1000-character art prompt and displays it in a text box.
  4. Renders the audio as a square Mel-spectrogram image with the **Geekatplay Orange Blue** colormap for AI training.
- **How to use**: Drag and drop `audio_to_spectrogram_and_prompt.json` directly onto the ComfyUI canvas.

#### Workflow 2: Spectrogram Image to Audio Reconstruction
- **File**: `workflows/spectrogram_to_audio.json`
- **What it does**:
  1. Loads a saved spectrogram PNG image.
  2. Reads the embedded audio parameters from JSON metadata.
  3. Uses Griffin-Lim (or Phase Inversion) to synthesize the original audio waveform.
  4. Saves the reconstructed sound as a `.wav` file.
- **How to use**: Drag and drop `spectrogram_to_audio.json` directly onto the ComfyUI canvas.

---

## 🧩 Custom Nodes Reference

| Node Name | Category | Function |
| :--- | :--- | :--- |
| **`GAP Load Audio`** | `Geekatplay Studio/Audio` | Loads WAV, MP3, FLAC, OGG, or M4A files into standard ComfyUI `AUDIO` tensors. |
| **`GAP Save Audio`** | `Geekatplay Studio/Audio` | Transposes waveform matrices and exports WAV audio files to `ComfyUI/output`. |
| **`GAP Audio To Spectrogram`** | `Geekatplay Studio/Spectrogram` | Converts audio into Mel-spectrogram images (512x512) or Phase-Encoded RGB images. |
| **`GAP Spectrogram To Audio`** | `Geekatplay Studio/Spectrogram` | Reconstructs audio from spectrogram images using Griffin-Lim or Phase Inversion. |
| **`GAP Music Analyser & Prompt`** | `Geekatplay Studio/Audio` | Extracts musical metrics (Key, BPM, Centroid, RMS) and generates ~1000-char prompts via Ollama/Fallback. |
| **`GAP Display Text Box`** | `Geekatplay Studio/Utility` | Displays generated prompt descriptions directly inside the ComfyUI UI canvas. |

---

## ❓ Troubleshooting & FAQ

### 1. `"ModuleNotFoundError: No module named 'librosa'"`
- **Fix**: Re-run `install_deps.bat` (on PC) or `install_deps.sh` (on Mac). Make sure you run it with the exact Python environment that ComfyUI uses (e.g. `python_embeded` if using ComfyUI Portable).

### 2. `"Ollama connection timed out / Ollama model not found"`
- **Fix**: Check that Ollama is running (`http://localhost:11434`). If using a custom model name, type it into the `ollama_model` field (e.g. `mistral`, `gemma2`). If Ollama is turned off, set `use_ollama` to `False` in the node to use the offline rules generator.

### 3. `"Where are the reconstructed audio files saved?"`
- **Fix**: Reconstructed WAV files are saved in your main `ComfyUI/output/` directory with the prefix `Geekatplay_Reconstructed`.

---

## 🏷️ License & Credits

Designed and developed for **Geekatplay Studio by Vladimir Chopine**.  
Visit [Geekatplay Studio](https://www.geekatplay.com) for tutorials, digital art resources, and 3D/AI workflows.
