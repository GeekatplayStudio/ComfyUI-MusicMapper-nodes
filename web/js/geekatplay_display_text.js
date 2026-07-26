import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Geekatplay.MusicMapper.DisplayTextBox",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeekatplayDisplayTextBox" || nodeData.name === "GeekatplayMusicAnalyser") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message && (message.text || message.string)) {
                    const textArr = message.text || message.string;
                    const textStr = Array.isArray(textArr) ? textArr.join("\n") : textArr;

                    let widget = this.widgets?.find((w) => w.name === "display_text" || w.name === "text" || w.name === "generated_prompt");
                    if (!widget) {
                        widget = this.addWidget("text", "display_text", textStr, () => {}, { multiline: true });
                    } else {
                        widget.value = textStr;
                    }

                    if (widget.inputEl) {
                        widget.inputEl.value = textStr;
                    }

                    if (this.size && (this.size[0] < 450 || this.size[1] < 200)) {
                        this.setSize([480, 260]);
                    }
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    }
});
