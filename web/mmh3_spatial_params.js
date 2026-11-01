// Front-end helper for MMH3SpatialSplitParams / LTX25SpatialSplitParams:
// auto-show/hide the two input groups depending on the tile_size_mode combo.
// Hidden widgets keep their stored values and serialize as usual - the server
// side already validates per mode, this is purely cosmetic.
import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set(["MMH3SpatialSplitParams", "LTX25SpatialSplitParams"]);
const MODE_WIDGET = "tile_size_mode";
const SPECIFIC_WIDGETS = ["tile_width", "tile_height"];
const GRID_WIDGETS = ["upscale_width", "upscale_height", "grid_rows", "grid_cols"];

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function setWidgetVisible(node, name, visible) {
    const w = findWidget(node, name);
    if (!w) return;
    if ("hidden" in w) {
        // modern ComfyUI frontend: proper hidden flag, excluded from layout
        w.hidden = !visible;
        return;
    }
    // fallback for older frontends: park the widget as "converted"
    if (visible) {
        if (w.type === "converted-widget" && w.origType) {
            w.type = w.origType;
            if (w.origComputeOptions) w.computeOptions = w.origComputeOptions;
        }
    } else if (w.type !== "converted-widget") {
        w.origType = w.type;
        w.origComputeOptions = w.computeOptions;
        w.type = "converted-widget";
        w.computeOptions = () => {};
    }
}

function applyMode(node) {
    const mode = findWidget(node, MODE_WIDGET);
    if (!mode) return false;
    const rowsCols = mode.value === "rows_cols";
    const isH3 = (node.comfyClass || node.type) === "MMH3SpatialSplitParams";
    const auto = isH3 && mode.value === "auto";
    for (const name of SPECIFIC_WIDGETS) setWidgetVisible(node, name, !rowsCols && !auto);
    for (const name of GRID_WIDGETS) setWidgetVisible(node, name, rowsCols && (!isH3 || !name.startsWith("upscale_")));
    if (isH3) setWidgetVisible(node, "token_budget", auto);
    if (node.onResize) node.onResize(node.size);
    app.graph.setDirtyCanvas(true, true);
    return true;
}

app.registerExtension({
    name: "MMH3UltimateUpscale.SpatialTileSizeMode",
    nodeCreated(node) {
        const cls = node.comfyClass || node.type;
        if (!TARGET_NODES.has(cls)) return;
        console.info("[MMH3-UltimateUpscale] tile_size_mode UI hook installed on", cls);

        const mode = findWidget(node, MODE_WIDGET);
        if (mode) {
            const origCallback = mode.callback;
            mode.callback = function () {
                const res = origCallback ? origCallback.apply(this, arguments) : undefined;
                applyMode(node);
                return res;
            };
        }
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const res = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            const values = arguments[0]?.widgets_values;
            if ((node.comfyClass || node.type) === "MMH3SpatialSplitParams" && Array.isArray(values)) {
                // LD saved joint controls immediately after overlap_blend, before
                // upstream added four sequential-only controls at those positions.
                if ((values.length === 15 || values.length === 16) && typeof values[14] === "boolean") {
                    const defaults = { masked_area_noise: 0, brightness_match: false, dynamic_fade: "off", dynamic_fade_min: 32 };
                    for (const [name, value] of Object.entries(defaults)) {
                        const widget = findWidget(node, name);
                        if (widget) widget.value = value;
                    }
                    const joint = findWidget(node, "joint_steps");
                    const budget = findWidget(node, "token_budget");
                    if (joint) joint.value = values[14];
                    if (budget && typeof values[15] === "number") budget.value = values[15];
                } else if (values.length === 18 && typeof values[14] === "number") {
                    // Workflows saved by upstream predate joint sampling and
                    // must continue using the sequential path when reopened.
                    const joint = findWidget(node, "joint_steps");
                    if (joint) joint.value = false;
                }
            }
            setTimeout(() => applyMode(node), 0);
            return res;
        };
        setTimeout(() => applyMode(node), 0);
    },
});
