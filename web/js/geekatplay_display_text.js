import { app } from "../../../scripts/app.js";

// Register Geekatplay Display Text Box Frontend Extension
app.registerExtension({
    name: "Geekatplay.MusicMapper.DisplayTextBox",
    async nodeCreated(node) {
        if (node.comfyClass === "GeekatplayDisplayTextBox") {
            // Ensure node has a multiline text display widget
            let widget = node.widgets?.find((w) => w.name === "display_text");
            if (!widget) {
                widget = node.addWidget("text", "display_text", "", () => {}, { multiline: true });
            }
            
            // Adjust node canvas size so long prompts (1000+ chars) are easy to read
            node.size = [450, 220];

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
                        node.setDirtyCanvas(true, true);
                    }
                }
            };
        }
    }
});
