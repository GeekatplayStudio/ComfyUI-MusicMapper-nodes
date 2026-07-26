import { app } from "../../../scripts/app.js";

// Register Geekatplay Display Text Box Frontend Extension
app.registerExtension({
    name: "Geekatplay.MusicMapper.DisplayTextBox",
    async nodeCreated(node) {
        if (node.comfyClass === "GeekatplayDisplayTextBox") {
            // Find or create multiline text widget
            let widget = node.widgets?.find((w) => w.name === "text" || w.name === "display_text");
            if (!widget) {
                widget = node.addWidget("text", "text", "", () => {}, { multiline: true });
            }
            
            // Set initial node size on canvas
            node.size = [460, 240];

            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                origOnExecuted?.apply(this, arguments);
                if (message) {
                    let textContent = "";
                    if (message.text) {
                        textContent = Array.isArray(message.text) ? message.text.join("\n") : message.text;
                    } else if (message.string) {
                        textContent = Array.isArray(message.string) ? message.string.join("\n") : message.string;
                    }
                    if (textContent) {
                        widget.value = textContent;
                        if (widget.inputEl) {
                            widget.inputEl.value = textContent;
                        }
                        node.setDirtyCanvas(true, true);
                    }
                }
            };
        }
    }
});
