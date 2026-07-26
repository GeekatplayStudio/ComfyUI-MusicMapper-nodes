import { app } from "../../../scripts/app.js";

// Register Geekatplay Display Text Box Extension using beforeRegisterNodeDef hook
app.registerExtension({
    name: "Geekatplay.MusicMapper.DisplayTextBox",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeekatplayDisplayTextBox" || nodeData.name === "GeekatplayPromptViewer") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                origOnExecuted?.apply(this, arguments);
                
                if (message) {
                    let textContent = "";
                    if (message.text) {
                        textContent = Array.isArray(message.text) ? message.text.join("\n") : message.text;
                    } else if (message.string) {
                        textContent = Array.isArray(message.string) ? message.string.join("\n") : message.string;
                    }
                    
                    if (textContent) {
                        // Find or add multiline text widget
                        let widget = this.widgets?.find((w) => w.name === "display_text" || w.name === "text");
                        if (!widget) {
                            widget = this.addWidget("text", "display_text", "", () => {}, { multiline: true });
                        }
                        
                        widget.value = textContent;
                        if (widget.inputEl) {
                            widget.inputEl.value = textContent;
                        }
                        
                        this.setSize([480, 260]);
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            };
        }
    }
});
