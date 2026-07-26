import { app } from "../../../scripts/app.js";

// Register Geekatplay OS File Browser Extension for Audio Loading
app.registerExtension({
    name: "Geekatplay.MusicMapper.LoadAudioBrowser",
    async nodeCreated(node) {
        if (node.comfyClass === "GeekatplayLoadAudio") {
            // Hidden HTML file input element for OS file browsing
            const fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length > 0) {
                        await uploadAudioFile(fileInput.files[0]);
                    }
                }
            });
            document.body.appendChild(fileInput);

            async function uploadAudioFile(file) {
                try {
                    const body = new FormData();
                    body.append("image", file); // ComfyUI upload endpoint expects 'image' key for uploads
                    body.append("overwrite", "true");
                    
                    const resp = await fetch("/upload/image", {
                        method: "POST",
                        body: body
                    });
                    
                    if (resp.status === 200) {
                        const data = await resp.json();
                        let path = data.name;
                        if (data.subfolder) path = data.subfolder + "/" + path;
                        
                        let widget = node.widgets?.find((w) => w.name === "audio_file");
                        if (widget) {
                            if (!widget.options.values.includes(path)) {
                                widget.options.values.push(path);
                            }
                            widget.value = path;
                        }
                        node.setDirtyCanvas(true, true);
                    } else {
                        console.error("[Geekatplay MusicMapper] Audio upload error status:", resp.status);
                    }
                } catch (error) {
                    console.error("[Geekatplay MusicMapper] Audio upload exception:", error);
                }
            }

            // Add interactive button to node UI
            node.addWidget("button", "📁 Browse OS File System...", "upload", () => {
                fileInput.click();
            });
        }
    }
});
