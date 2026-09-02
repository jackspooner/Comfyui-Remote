import { app } from "../../scripts/app.js";

const INPUT_CLASS = "CutleryWorkflowInput";
const OUTPUT_CLASS = "CutleryWorkflowOutput";
const MAX_PORTS = 64;
const VALUE_NAMES = Array.from({ length: MAX_PORTS }, (_, index) => `value_${index + 1}`);
const SOCKET_TYPES = {
  image: "IMAGE",
  mask: "MASK",
  audio: "AUDIO",
  video: "VIDEO",
  latent: "LATENT",
  conditioning: "CONDITIONING",
  string: "STRING",
  text: "STRING",
  int: "INT",
  integer: "INT",
  float: "FLOAT",
  number: "FLOAT",
  bool: "BOOLEAN",
  boolean: "BOOLEAN",
  json: "*",
  any: "*",
  "*": "*",
  cutlery_lora_chain: "CUTLERY_LORA_CHAIN",
};

function widgetByName(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

function parsePorts(node) {
  const raw = widgetByName(node, "ports_json")?.value ?? "[]";
  try {
    const parsed = JSON.parse(String(raw || "[]"));
    const records = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.ports) ? parsed.ports : [];
    return records.slice(0, MAX_PORTS).map((record, index) => {
      const name = String(record?.name ?? "").trim();
      if (!name) throw new Error(`Port ${index + 1} needs a name.`);
      const declaredType = String(record?.type ?? "any").trim().toLowerCase();
      return { name, type: SOCKET_TYPES[declaredType] ?? declaredType.toUpperCase() };
    });
  } catch (_error) {
    return [];
  }
}

function removeSlot(node, direction, index) {
  if (direction === "input" && typeof node.removeInput === "function") return node.removeInput(index);
  if (direction === "output" && typeof node.removeOutput === "function") return node.removeOutput(index);
  const slots = direction === "input" ? node.inputs : node.outputs;
  slots?.splice(index, 1);
}

function reconcileSlots(node, direction, ports) {
  const slots = direction === "input" ? node.inputs : node.outputs;
  const add = direction === "input" ? node.addInput : node.addOutput;
  if (!Array.isArray(slots) || typeof add !== "function") return;

  for (let index = slots.length - 1; index >= ports.length; index -= 1) {
    const slot = slots[index];
    if (slot?.__cutleryWorkflowBoundary || VALUE_NAMES.includes(slot?.name)) removeSlot(node, direction, index);
  }
  for (let index = 0; index < ports.length; index += 1) {
    let slot = slots[index];
    if (!slot) {
      add.call(node, ports[index].name, ports[index].type);
      slot = slots[index];
    }
    if (!slot) continue;
    slot.name = ports[index].name;
    slot.type = ports[index].type;
    slot.label = `${ports[index].name} (${ports[index].type})`;
    slot.localized_name = slot.label;
    slot.__cutleryWorkflowBoundary = true;
  }
}

function refreshBoundaryPorts(node) {
  const ports = parsePorts(node);
  if (node.__cutleryWorkflowBoundaryDirection === "input") {
    reconcileSlots(node, "input", ports);
    reconcileSlots(node, "output", ports);
  } else {
    reconcileSlots(node, "input", ports);
  }
  const size = node.computeSize?.();
  if (size) node.setSize?.(size);
  node.graph?.setDirtyCanvas?.(true, true);
}

function scheduleRefresh(node) {
  window.clearTimeout(node.__cutleryWorkflowBoundaryTimer);
  node.__cutleryWorkflowBoundaryTimer = window.setTimeout(() => refreshBoundaryPorts(node), 0);
}

function stripPlaceholderSlots(nodeData, direction) {
  if (direction === "input") {
    const optional = nodeData.input?.optional;
    if (optional && !Array.isArray(optional)) {
      for (const name of VALUE_NAMES) delete optional[name];
    }
    if (Array.isArray(nodeData.input_order?.optional)) {
      nodeData.input_order.optional = nodeData.input_order.optional.filter((name) => !VALUE_NAMES.includes(name));
    }
  }
  if (direction === "output") {
    for (const key of ["output", "output_name", "output_is_list"]) {
      if (Array.isArray(nodeData[key])) nodeData[key] = [];
    }
  }
}

function installBoundaryNode(nodeType, direction) {
  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function onCutleryWorkflowBoundaryCreated(...args) {
    const result = originalCreated?.apply(this, args);
    this.__cutleryWorkflowBoundaryDirection = direction;
    const widget = widgetByName(this, "ports_json");
    if (widget && !widget.__cutleryWorkflowBoundaryInstalled) {
      const originalCallback = widget.callback;
      widget.callback = (...callbackArgs) => {
        const callbackResult = originalCallback?.apply(widget, callbackArgs);
        scheduleRefresh(this);
        return callbackResult;
      };
      widget.__cutleryWorkflowBoundaryInstalled = true;
    }
    scheduleRefresh(this);
    return result;
  };
}

app.registerExtension({
  name: "Cutlery.WorkflowBoundaries",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === INPUT_CLASS) {
      stripPlaceholderSlots(nodeData, "input");
      stripPlaceholderSlots(nodeData, "output");
      if (!nodeType.prototype.__cutleryWorkflowBoundaryInstalled) {
        installBoundaryNode(nodeType, "input");
        nodeType.prototype.__cutleryWorkflowBoundaryInstalled = true;
      }
    }
    if (nodeData?.name === OUTPUT_CLASS) {
      stripPlaceholderSlots(nodeData, "input");
      if (!nodeType.prototype.__cutleryWorkflowBoundaryInstalled) {
        installBoundaryNode(nodeType, "output");
        nodeType.prototype.__cutleryWorkflowBoundaryInstalled = true;
      }
    }
  },
  nodeCreated(node) {
    if ([INPUT_CLASS, OUTPUT_CLASS].includes(node?.comfyClass ?? node?.constructor?.comfyClass)) scheduleRefresh(node);
  },
  loadedGraphNode(node) {
    if ([INPUT_CLASS, OUTPUT_CLASS].includes(node?.comfyClass ?? node?.constructor?.comfyClass)) scheduleRefresh(node);
  },
});
