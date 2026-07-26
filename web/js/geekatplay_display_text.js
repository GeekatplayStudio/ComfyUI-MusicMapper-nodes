import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Register Geekatplay Display Text Box Extension
app.registerExtension({
    name: "Geekatplay.MusicMapper.DisplayTextBox",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeekatplayDisplayTextBox") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                if (message && message.string) {
                    const text = Array.isArray(message.string) ? message.string.join("\n") : String(message.string);
                    
                    // Look for existing text widget or create a multiline string widget
                    let widget = this.widgets?.find((w) => w.name === "display_text" || w.name === "text");
                    if (!widget) {
                        widget = ComfyWidgets["STRING"](this, "display_text", ["STRING", { multiline: true }], app).widget;
                    }
                    
                    widget.value = text;
                    this.size[1] = Math.max(this.size[1], 220); // Expand box height if needed
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
});
