import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MODEL_NODE_CLASS = "CutleryRemoteModelName";
const GROUP_EXECUTOR_CLASS = "CutleryRemoteGroupExecutor";
const GROUP_VALUE_EXECUTOR_CLASS = "CutleryRemoteGroupValueExecutor";
const GROUP_EXECUTOR_CLASSES = new Set([GROUP_EXECUTOR_CLASS, GROUP_VALUE_EXECUTOR_CLASS]);
const EXTENSION_NAME = "Cutlery.RemoteModels";
const MODEL_ENDPOINT = "/cutlery/remote/models";
const NODE_DEFINITIONS_ENDPOINT = "/cutlery/remote/proxy/node-definitions";
const MAX_PORTS = 64;
const DEFINITION_CACHE_TTL_MS = 30_000;
const GRAPH_REFRESH_DELAY_MS = 120;
const COLLAPSED_NODE_HEIGHT = 30;
const VALUE_NAMES = Array.from({ length: MAX_PORTS }, (_, index) => `value_${index + 1}`);
const EXECUTOR_TRANSPORT_WIDGETS = [
  "remote_base_url",
  "remote_workflow_json",
  "input_ports_json",
  "output_ports_json",
  "cache_policy",
];
const SOCKET_TYPES = {
  image: "IMAGE",
  mask: "MASK",
  audio: "AUDIO",
  video: "VIDEO",
  latent: "LATENT",
  conditioning: "CONDITIONING",
  string: "STRING",
  int: "INT",
  integer: "INT",
  float: "FLOAT",
  number: "FLOAT",
  bool: "BOOLEAN",
  boolean: "BOOLEAN",
  json: "*",
  "*": "*",
  cutlery_lora_chain: "CUTLERY_LORA_CHAIN",
};
const TARGET_RE = /^([A-Za-z0-9_.-]+):([1-9][0-9]{0,4})$/;
const CURLY_TARGET_RE = /^([A-Za-z0-9_.-]+):\{([1-9][0-9]{0,4})\}$/;
const CUTLERY_TARGET_RE = /^cutlery:\/\/([A-Za-z0-9_.-]+):([1-9][0-9]{0,4})$/;
const CUTLERY_ALIAS_RE = /^cutlery:\/\/([A-Za-z0-9_.-]+)$/;

let fetchHookInstalled = false;
let graphRefreshInstalled = false;
let graphRefreshTimer = null;
let refreshGeneration = 0;

const definitionCache = new Map();
const definitionRequests = new Map();
const modelChoiceCache = new Map();
const modelChoiceRequests = new Map();
const widgetOverlayStates = new WeakMap();
const uploadActionStates = new WeakMap();
const overlayNodes = new Set();
const registryAdapters = new Map();

function nodeClassType(node) {
  return String(node?.comfyClass ?? node?.constructor?.comfyClass ?? node?.constructor?.__cutleryClassName ?? node?.type ?? "").trim();
}

function isRemoteModelNode(node) {
  return nodeClassType(node) === MODEL_NODE_CLASS;
}

function widgetByName(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

function hideWidget(node, name) {
  const widget = widgetByName(node, name);
  if (!widget) {
    return;
  }
  widget.hidden = true;
  widget.serialize = true;
  widget.disabled = true;
  widget.computeSize = () => [0, -4];
}

function hideExecutorTransportWidgets(node) {
  for (const name of EXECUTOR_TRANSPORT_WIDGETS) {
    hideWidget(node, name);
  }
}

function parseRemoteTargetTitle(title) {
  const text = String(title ?? "").trim();
  const aliasMatch = text.match(CUTLERY_ALIAS_RE);
  if (aliasMatch) {
    return `cutlery://${aliasMatch[1]}`;
  }
  const match = text.match(TARGET_RE) ?? text.match(CURLY_TARGET_RE) ?? text.match(CUTLERY_TARGET_RE);
  if (!match) {
    return null;
  }
  const port = Number.parseInt(match[2], 10);
  if (!Number.isFinite(port) || port < 1 || port > 65535) {
    return null;
  }
  return `${match[1]}:${port}`;
}

function normalizeBounds(value) {
  if (!value || typeof value !== "object" || Number(value.length ?? 0) < 4) {
    return null;
  }
  const bounds = Array.from(value).slice(0, 4).map((item) => Number(item));
  return bounds.every(Number.isFinite) ? bounds : null;
}

function boundsFromMethod(subject, methodName) {
  if (typeof subject?.[methodName] !== "function") {
    return null;
  }
  try {
    const output = [Number.NaN, Number.NaN, Number.NaN, Number.NaN];
    return normalizeBounds(subject[methodName](output)) ?? normalizeBounds(output);
  } catch (_error) {
    return null;
  }
}

function nodeBounds(node) {
  const pos = node?.pos ?? [0, 0];
  const size = node?.size ?? [0, 0];
  const positionalBounds = [
    Number(pos[0] ?? 0),
    Number(pos[1] ?? 0),
    Number(size[0] ?? 0),
    node?.flags?.collapsed ? COLLAPSED_NODE_HEIGHT : Number(size[1] ?? 0),
  ];
  for (const candidate of [
    node?.boundingRect,
    node?.bounding,
    node?._bounding,
    boundsFromMethod(node, "getBounding"),
    boundsFromMethod(node, "getBoundingRect"),
  ]) {
    const normalized = normalizeBounds(candidate);
    if (
      normalized &&
      (normalized[2] > 0 ||
        normalized[3] > 0 ||
        (positionalBounds[2] <= 0 && positionalBounds[3] <= 0))
    ) {
      return normalized;
    }
  }
  return positionalBounds;
}

function groupBounds(group) {
  for (const candidate of [
    group?.boundingRect,
    group?.bounding,
    group?._bounding,
    boundsFromMethod(group, "getBounding"),
    boundsFromMethod(group, "getBoundingRect"),
  ]) {
    const normalized = normalizeBounds(candidate);
    if (normalized) {
      return normalized;
    }
  }
  if (typeof group?.serialize === "function") {
    try {
      return normalizeBounds(group.serialize()?.bounding);
    } catch (_error) {
      return null;
    }
  }
  return null;
}

function boundsContain(outer, inner) {
  if (!outer || !inner) {
    return false;
  }
  const [outerX, outerY, outerWidth, outerHeight] = outer;
  const [innerX, innerY, innerWidth, innerHeight] = inner;
  return (
    innerX >= outerX &&
    innerY >= outerY &&
    innerX + innerWidth <= outerX + outerWidth &&
    innerY + innerHeight <= outerY + outerHeight
  );
}

function boundsOverlap(first, second) {
  const [firstX, firstY, firstWidth, firstHeight] = first;
  const [secondX, secondY, secondWidth, secondHeight] = second;
  return (
    firstX < secondX + secondWidth &&
    firstX + firstWidth > secondX &&
    firstY < secondY + secondHeight &&
    firstY + firstHeight > secondY
  );
}

function graphForNode(node) {
  return node?.graph ?? app.graph ?? app.canvas?.graph ?? null;
}

function nodeId(node) {
  return String(node?.id ?? "");
}

function subgraphDefinitions(graph) {
  const definitions = graph?._subgraphs;
  if (definitions instanceof Map) {
    return definitions;
  }
  const byId = new Map();
  const values = Array.isArray(definitions) ? definitions : Object.values(definitions ?? {});
  for (const definition of values) {
    const id = String(definition?.id ?? "").trim();
    if (id) {
      byId.set(id, definition);
    }
  }
  return byId;
}

function subgraphForNode(node, rootGraph) {
  if (node?.subgraph) {
    return node.subgraph;
  }
  return subgraphDefinitions(rootGraph).get(nodeClassType(node)) ?? null;
}

function graphContexts(rootGraph) {
  if (!rootGraph) {
    return [];
  }
  const contexts = [];
  const visit = (graph, promptPrefix, ancestors) => {
    contexts.push({ graph, promptPrefix });
    for (const node of graph?._nodes ?? []) {
      const subgraph = subgraphForNode(node, rootGraph);
      if (!subgraph || ancestors.has(subgraph)) {
        continue;
      }
      visit(
        subgraph,
        `${promptPrefix}${nodeId(node)}:`,
        new Set([...ancestors, subgraph]),
      );
    }
  };
  visit(rootGraph, "", new Set([rootGraph]));
  return contexts;
}

function inspectRemoteGroupMembership(graph) {
  const groupInfos = [];
  const membershipByNode = new Map();
  const errors = [];
  const canvasNodeByPromptId = new Map();

  for (const context of graphContexts(graph)) {
    const contextGroups = [];
    for (const group of context.graph?._groups ?? context.graph?.groups ?? []) {
      const target = parseRemoteTargetTitle(group?.title);
      if (target) {
        const groupInfo = {
          graph: context.graph,
          group,
          target,
          promptPrefix: context.promptPrefix,
          promptIdByNode: new Map(),
          insideNodes: new Set(),
          insideIds: new Set(),
          canvasNodeByPromptId,
        };
        contextGroups.push(groupInfo);
        groupInfos.push(groupInfo);
      }
    }

    for (const node of context.graph?._nodes ?? []) {
      const id = nodeId(node);
      const promptId = id ? `${context.promptPrefix}${id}` : "";
      if (promptId) {
        canvasNodeByPromptId.set(promptId, node);
      }
      const bounds = nodeBounds(node);
      const matches = contextGroups.filter((groupInfo) => boundsContain(groupBounds(groupInfo.group), bounds));
      const partialMatches = contextGroups.filter((groupInfo) => {
        const bounding = groupBounds(groupInfo.group);
        return bounding && boundsOverlap(bounding, bounds) && !boundsContain(bounding, bounds);
      });
      if (partialMatches.length) {
        const message = `Node ${id || nodeClassType(node) || "(unknown)"} partially overlaps a Cutlery remote group. Move it fully inside or outside.`;
        errors.push({ node, promptId, message });
        continue;
      }
      if (matches.length > 1) {
        const message = `Node ${id || nodeClassType(node) || "(unknown)"} is inside multiple Cutlery remote groups.`;
        errors.push({ node, promptId, message });
        continue;
      }
      const groupInfo = matches[0];
      if (!groupInfo) {
        continue;
      }
      membershipByNode.set(node, groupInfo);
      groupInfo.insideNodes.add(node);
      if (promptId) {
        groupInfo.promptIdByNode.set(node, promptId);
        groupInfo.insideIds.add(promptId);
      }
    }
  }

  return {
    groups: groupInfos.filter((groupInfo) => groupInfo.insideNodes.size > 0),
    membershipByNode,
    errors,
    canvasNodeByPromptId,
  };
}

function promptNodeForCanvasNode(prompt, node, promptId = null) {
  const id = String(promptId ?? nodeId(node));
  if (!id || !Object.prototype.hasOwnProperty.call(prompt ?? {}, id)) {
    return null;
  }
  const promptNode = prompt[id];
  if (!promptNode || typeof promptNode !== "object" || Array.isArray(promptNode)) {
    return null;
  }
  return promptNode;
}

function promptNodeMatchesCanvasNode(prompt, node, promptId = null) {
  const promptNode = promptNodeForCanvasNode(prompt, node, promptId);
  if (!promptNode) {
    return false;
  }
  const promptClassType = String(promptNode.class_type ?? "").trim();
  const canvasClassType = nodeClassType(node);
  return Boolean(promptClassType && canvasClassType && promptClassType === canvasClassType);
}

function remoteGroupInspectionForPrompt(graph, prompt) {
  const inspection = inspectRemoteGroupMembership(graph);
  if (!prompt || typeof prompt !== "object" || Array.isArray(prompt)) {
    return {
      groups: [],
      membershipByNode: inspection.membershipByNode,
      errors: [],
    };
  }

  const groups = [];
  for (const groupInfo of inspection.groups) {
    const intersectingNodes = [...groupInfo.insideNodes].filter((node) =>
      Boolean(promptNodeForCanvasNode(prompt, node, groupInfo.promptIdByNode.get(node))),
    );
    if (!intersectingNodes.length) {
      continue;
    }
    if (
      intersectingNodes.some((node) =>
        !promptNodeMatchesCanvasNode(prompt, node, groupInfo.promptIdByNode.get(node)),
      )
    ) {
      continue;
    }
    const insideNodes = new Set(intersectingNodes);
    const promptIdByNode = new Map(
      [...insideNodes].map((node) => [node, groupInfo.promptIdByNode.get(node)]),
    );
    groups.push({
      ...groupInfo,
      insideNodes,
      promptIdByNode,
      insideIds: new Set(promptIdByNode.values()),
    });
  }

  return {
    groups,
    membershipByNode: inspection.membershipByNode,
    errors: inspection.errors.filter(({ node, promptId }) =>
      promptNodeMatchesCanvasNode(prompt, node, promptId),
    ),
    canvasNodeByPromptId: inspection.canvasNodeByPromptId,
  };
}

function remoteGroupForNode(node, inspection = null) {
  return (inspection ?? inspectRemoteGroupMembership(graphForNode(node))).membershipByNode.get(node) ?? null;
}

function remoteTargetForNode(node, inspection = null) {
  return remoteGroupForNode(node, inspection)?.target ?? null;
}

function typedUnique(values) {
  const output = [];
  for (const value of Array.isArray(values) ? values : []) {
    if (!output.some((item) => Object.is(item, value))) {
      output.push(value);
    }
  }
  return output;
}

function selectedValueIsAvailable(widget, values) {
  return values.some((value) => Object.is(value, widget?.value));
}

function captureLocalWidgetState(widget, target) {
  let state = widgetOverlayStates.get(widget);
  if (state && state.target !== target) {
    restoreWidgetOptions(widget);
    state = null;
  }
  if (!state) {
    state = {
      target,
      localType: widget.type,
      hadValues: Object.prototype.hasOwnProperty.call(widget.options ?? {}, "values"),
      localValues: Array.isArray(widget.options?.values) ? widget.options.values.slice() : widget.options?.values,
      appliedValues: null,
    };
    widgetOverlayStates.set(widget, state);
    return state;
  }
  if (widget.options?.values !== state.appliedValues) {
    state.localType = widget.type;
    state.hadValues = Object.prototype.hasOwnProperty.call(widget.options ?? {}, "values");
    state.localValues = Array.isArray(widget.options?.values) ? widget.options.values.slice() : widget.options?.values;
  }
  return state;
}

function applyRemoteWidgetOptions(node, widget, target, values, metadata = {}) {
  if (!widget) {
    return;
  }
  const state = captureLocalWidgetState(widget, target);
  const choices = typedUnique(values);
  widget.type = "combo";
  widget.options = widget.options ?? {};
  widget.options.values = choices;
  widget.__cutleryRemoteTarget = target;
  widget.__cutleryRemoteOptionsError = metadata.error ?? null;
  widget.__cutleryRemoteUploadBacked = Boolean(metadata.uploadBacked);
  state.appliedValues = choices;
  overlayNodes.add(node);
}

function restoreWidgetOptions(widget) {
  const state = widgetOverlayStates.get(widget);
  if (!state) {
    return;
  }
  widget.type = state.localType;
  widget.options = widget.options ?? {};
  if (state.hadValues) {
    widget.options.values = Array.isArray(state.localValues) ? state.localValues.slice() : state.localValues;
  } else {
    delete widget.options.values;
  }
  delete widget.__cutleryRemoteTarget;
  delete widget.__cutleryRemoteOptionsError;
  delete widget.__cutleryRemoteUploadBacked;
  widgetOverlayStates.delete(widget);
}

function isUploadActionWidget(widget) {
  const label = `${widget?.name ?? ""} ${widget?.label ?? ""}`.toLowerCase();
  return label.includes("upload") && (widget?.type === "button" || typeof widget?.callback === "function");
}

function restoreRemoteUploadActions(node) {
  for (const widget of node?.widgets ?? []) {
    const state = uploadActionStates.get(widget);
    if (!state) {
      continue;
    }
    widget.disabled = state.disabled;
    widget.callback = state.callback;
    if (state.hadOptionDisabled) {
      widget.options = widget.options ?? {};
      widget.options.disabled = state.optionDisabled;
    } else if (widget.options) {
      delete widget.options.disabled;
    }
    if (widget.inputEl) {
      widget.inputEl.disabled = state.inputDisabled;
    }
    delete widget.__cutleryRemoteUploadDisabled;
    uploadActionStates.delete(widget);
  }
}

function disableRemoteUploadActions(node, target) {
  for (const widget of node?.widgets ?? []) {
    if (!isUploadActionWidget(widget)) {
      continue;
    }
    let state = uploadActionStates.get(widget);
    if (state?.target !== target) {
      if (state) {
        restoreRemoteUploadActions(node);
      }
      state = null;
    }
    if (!state) {
      state = {
        target,
        disabled: widget.disabled,
        callback: widget.callback,
        hadOptionDisabled: Object.prototype.hasOwnProperty.call(widget.options ?? {}, "disabled"),
        optionDisabled: widget.options?.disabled,
        inputDisabled: widget.inputEl?.disabled,
      };
      uploadActionStates.set(widget, state);
    }
    widget.disabled = true;
    widget.options = widget.options ?? {};
    widget.options.disabled = true;
    if (widget.inputEl) {
      widget.inputEl.disabled = true;
    }
    widget.__cutleryRemoteUploadDisabled = true;
    widget.callback = function cutleryRemoteUploadDisabled() {
      return undefined;
    };
    overlayNodes.add(node);
  }
}

function restoreRemoteOptionsForNode(node) {
  for (const widget of node?.widgets ?? []) {
    restoreWidgetOptions(widget);
  }
  restoreRemoteUploadActions(node);
  delete node.__cutleryRemoteDefinition;
  overlayNodes.delete(node);
}

function ensureRemoteStatusRenderer(node) {
  if (!node || node.__cutleryRemoteStatusRendererInstalled) {
    return;
  }
  const original = node.onDrawForeground;
  node.onDrawForeground = function drawCutleryRemoteStatus(context, ...args) {
    const result = original?.call(this, context, ...args);
    const status = this.__cutleryRemoteOptionsStatus;
    if (!status?.message || !context?.save || this.flags?.collapsed) {
      return result;
    }
    const width = Math.max(80, Number(this.size?.[0] ?? 180));
    context.save();
    context.fillStyle = status.level === "error" ? "rgba(153, 27, 27, 0.94)" : "rgba(146, 64, 14, 0.94)";
    context.fillRect(0, 0, width, 18);
    context.fillStyle = "#ffffff";
    context.font = "11px sans-serif";
    context.textBaseline = "middle";
    const prefix = status.level === "error" ? "Remote error: " : "Remote: ";
    const availableWidth = Math.max(20, width - 10);
    let text = `${prefix}${status.message}`;
    while (text.length > 8 && context.measureText?.(text)?.width > availableWidth) {
      text = `${text.slice(0, -2)}…`;
    }
    context.fillText(text, 5, 9);
    context.restore();
    return result;
  };
  node.__cutleryRemoteStatusRendererInstalled = true;
}

function setNodeRemoteStatus(node, target, errors = [], warnings = []) {
  ensureRemoteStatusRenderer(node);
  const errorMessages = [...new Set(errors.filter(Boolean).map(String))];
  const warningMessages = [...new Set(warnings.filter(Boolean).map(String))];
  const level = errorMessages.length ? "error" : warningMessages.length ? "warning" : null;
  const message = errorMessages[0] ?? warningMessages[0] ?? null;
  node.__cutleryRemoteTarget = target ?? null;
  node.__cutleryRemoteOptionsError = errorMessages.join("\n") || null;
  node.__cutleryRemoteOptionsWarning = warningMessages.join("\n") || null;
  node.__cutleryRemoteOptionsStatus = message ? { level, message } : null;
  graphForNode(node)?.setDirtyCanvas?.(true, true);
}

function clearNodeRemoteStatus(node) {
  delete node.__cutleryRemoteTarget;
  delete node.__cutleryRemoteOptionsError;
  delete node.__cutleryRemoteOptionsWarning;
  delete node.__cutleryRemoteOptionsStatus;
  graphForNode(node)?.setDirtyCanvas?.(true, true);
}

function setRemoteTargetWidget(node, target) {
  const widget = widgetByName(node, "remote_target");
  if (widget) {
    widget.value = target ?? "";
  }
}

function normalizeSocketType(type) {
  const text = String(type ?? "*").trim();
  return SOCKET_TYPES[text.toLowerCase()] ?? text.toUpperCase();
}

function parsePortsFromWidget(node, widgetName) {
  const raw = widgetByName(node, widgetName)?.value ?? "[]";
  try {
    const parsed = JSON.parse(String(raw || "[]"));
    const records = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.ports) ? parsed.ports : [];
    return records
      .map((record, index) => ({
        name: String(record?.name ?? `value_${index + 1}`).trim() || `value_${index + 1}`,
        type: normalizeSocketType(record?.type ?? "*"),
      }))
      .slice(0, MAX_PORTS);
  } catch (_error) {
    return [];
  }
}

function stripInitialExecutorPorts(nodeData) {
  const optional = nodeData.input?.optional;
  const optionalOrder = nodeData.input_order?.optional;
  if (optional && !Array.isArray(optional)) {
    for (const name of VALUE_NAMES) {
      delete optional[name];
    }
  }
  if (Array.isArray(optionalOrder)) {
    nodeData.input_order.optional = optionalOrder.filter((name) => !VALUE_NAMES.includes(name));
  }
  for (const key of ["output", "output_name", "output_is_list"]) {
    if (Array.isArray(nodeData[key])) {
      nodeData[key] = [];
    }
  }
}

function removeSlot(node, direction, index) {
  if (direction === "input" && typeof node.removeInput === "function") {
    node.removeInput(index);
    return;
  }
  if (direction === "output" && typeof node.removeOutput === "function") {
    node.removeOutput(index);
    return;
  }
  const slots = direction === "input" ? node.inputs : node.outputs;
  slots?.splice(index, 1);
}

function ensureExecutorSlots(node, direction, ports) {
  const slots = direction === "input" ? node.inputs : node.outputs;
  const add = direction === "input" ? node.addInput : node.addOutput;
  if (!Array.isArray(slots) || typeof add !== "function") {
    return;
  }
  for (let index = slots.length - 1; index >= ports.length; index -= 1) {
    const slot = slots[index];
    if (slot?.__cutleryRemoteExecutor || VALUE_NAMES.includes(slot?.name)) {
      removeSlot(node, direction, index);
    }
  }
  for (let index = 0; index < ports.length; index += 1) {
    let slot = slots[index];
    if (!slot) {
      add.call(node, ports[index].name, ports[index].type);
      slot = slots[index];
    }
    if (slot) {
      slot.name = ports[index].name;
      slot.type = ports[index].type;
      slot.label = `${ports[index].name} (${ports[index].type})`;
      slot.localized_name = slot.label;
      slot.__cutleryRemoteExecutor = true;
    }
  }
}

function refreshExecutorPorts(node) {
  hideExecutorTransportWidgets(node);
  ensureExecutorSlots(node, "input", parsePortsFromWidget(node, "input_ports_json"));
  ensureExecutorSlots(node, "output", parsePortsFromWidget(node, "output_ports_json"));
  const size = node.computeSize?.();
  if (size) {
    node.setSize?.(size);
  }
  graphForNode(node)?.setDirtyCanvas?.(true, true);
}

function scheduleRefreshExecutorPorts(node) {
  window.clearTimeout(node.__cutleryRemoteExecutorTimer);
  node.__cutleryRemoteExecutorTimer = window.setTimeout(() => refreshExecutorPorts(node), 0);
}

function cacheKey(target, value) {
  return `${target}\u0000${value}`;
}

function cachedValue(cache, key) {
  const record = cache.get(key);
  if (!record || record.expiresAt <= Date.now()) {
    cache.delete(key);
    return undefined;
  }
  return record.value;
}

async function fetchRemoteNodeDefinitions(target, classTypes, { force = false } = {}) {
  const uniqueClassTypes = [...new Set(classTypes.filter(Boolean))].sort();
  const results = {};
  const missing = [];
  for (const classType of uniqueClassTypes) {
    const key = cacheKey(target, classType);
    const cached = force ? undefined : cachedValue(definitionCache, key);
    if (cached === undefined) {
      missing.push(classType);
    } else {
      results[classType] = cached;
    }
  }
  if (!missing.length) {
    return results;
  }

  const requestKey = cacheKey(target, missing.join("\u0001"));
  let request = definitionRequests.get(requestKey);
  if (!request) {
    request = (async () => {
      const response = await api.fetchApi(NODE_DEFINITIONS_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({ target, class_types: missing }),
      });
      const payload = await response.json().catch(() => ({}));
      if (response.ok === false || payload?.ok === false) {
        throw new Error(payload.error || payload.message || `${response.status ?? ""} ${response.statusText ?? ""}`.trim());
      }
      const nodes = payload?.nodes && typeof payload.nodes === "object" ? payload.nodes : {};
      const expiresAt = Date.now() + DEFINITION_CACHE_TTL_MS;
      const fetched = {};
      for (const classType of missing) {
        fetched[classType] = nodes[classType] ?? {
          available: false,
          error: `Node class ${classType} is not installed on ${target}.`,
          inputs: {},
        };
        definitionCache.set(cacheKey(target, classType), { value: fetched[classType], expiresAt });
      }
      return fetched;
    })().finally(() => definitionRequests.delete(requestKey));
    definitionRequests.set(requestKey, request);
  }
  return { ...results, ...(await request) };
}

async function fetchModelChoices(target, modelType, { force = false } = {}) {
  const key = cacheKey(target, modelType || "checkpoints");
  const cached = force ? undefined : cachedValue(modelChoiceCache, key);
  if (cached !== undefined) {
    return cached;
  }
  let request = modelChoiceRequests.get(key);
  if (!request) {
    request = (async () => {
      const params = new URLSearchParams({
        target,
        model_type: modelType || "checkpoints",
        include_hashes: "0",
      });
      const response = await api.fetchApi(`${MODEL_ENDPOINT}?${params.toString()}`, { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (response.ok === false || payload?.ok === false) {
        throw new Error(payload.error || payload.message || `${response.status ?? ""} ${response.statusText ?? ""}`.trim());
      }
      const choices = typedUnique(payload.models ?? payload[payload.model_type] ?? []);
      modelChoiceCache.set(key, { value: choices, expiresAt: Date.now() + DEFINITION_CACHE_TTL_MS });
      return choices;
    })().finally(() => modelChoiceRequests.delete(key));
    modelChoiceRequests.set(key, request);
  }
  return request;
}

function remoteComboInputs(nodeDefinition) {
  const records = {};
  const inputOptions = nodeDefinition?.input_options;
  if (inputOptions && typeof inputOptions === "object" && !Array.isArray(inputOptions)) {
    for (const [inputName, input] of Object.entries(inputOptions)) {
      if (input?.kind === "combo") {
        records[inputName] = input;
      }
    }
    return records;
  }

  const inputs = nodeDefinition?.inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return records;
  }
  const sections = ["required", "optional", "hidden"].some((name) => inputs[name] && typeof inputs[name] === "object")
    ? [inputs.required, inputs.optional, inputs.hidden]
    : [inputs];
  for (const section of sections) {
    for (const [inputName, input] of Object.entries(section ?? {})) {
      if (input?.kind === "combo") {
        records[inputName] = input;
      }
    }
  }
  return records;
}

function remoteInputRecords(nodeDefinition) {
  const records = {};
  const inputs = nodeDefinition?.inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return records;
  }
  const sections = ["required", "optional", "hidden"].some((name) => inputs[name] && typeof inputs[name] === "object")
    ? [inputs.required, inputs.optional, inputs.hidden]
    : [inputs];
  for (const section of sections) {
    for (const [inputName, input] of Object.entries(section ?? {})) {
      if (input && typeof input === "object" && !Array.isArray(input)) {
        records[inputName] = input;
      }
    }
  }
  return records;
}

function declaredLocalComboInputNames(node) {
  const names = node?.constructor?.prototype?.__cutleryLocalComboInputNames ?? node?.constructor?.__cutleryLocalComboInputNames;
  return names instanceof Set ? names : new Set();
}

function adapterMatchesNode(adapter, node) {
  if (typeof adapter?.matches === "function") {
    return Boolean(adapter.matches(node));
  }
  const classTypes = adapter?.classTypes instanceof Set ? adapter.classTypes : new Set(adapter?.classTypes ?? []);
  return classTypes.has(nodeClassType(node));
}

function registryAdaptersForNode(node) {
  return [...registryAdapters.values()].filter((adapter) => adapterMatchesNode(adapter, node));
}

function adapterManagedInputs(adapter, node) {
  const value = typeof adapter?.managedInputs === "function" ? adapter.managedInputs(node) : adapter?.managedInputs;
  return new Set(Array.isArray(value) || value instanceof Set ? value : []);
}

function registryManagedInputNames(node) {
  const names = new Set();
  for (const adapter of registryAdaptersForNode(node)) {
    for (const inputName of adapterManagedInputs(adapter, node)) {
      names.add(String(inputName));
    }
  }
  return names;
}

function restoreRegistryAdapters(node) {
  for (const adapter of registryAdaptersForNode(node)) {
    try {
      adapter.restore?.(node);
    } catch (error) {
      console.warn(`[Cutlery Remote Models] Failed to restore registry adapter ${adapter.id}`, error);
    }
  }
}

function widgetValueFingerprint(value) {
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function snapshotWidgetValues(node) {
  return new Map((node?.widgets ?? []).map((widget) => [widget, widgetValueFingerprint(widget.value)]));
}

function changedWidgetNames(node, snapshot) {
  if (!(snapshot instanceof Map)) {
    return [];
  }
  return (node?.widgets ?? [])
    .filter((widget) => snapshot.has(widget) && snapshot.get(widget) !== widgetValueFingerprint(widget.value))
    .map((widget) => String(widget.name ?? "(unnamed)"));
}

async function refreshRegistryAdapters(node, target, { force = false, preflight = false } = {}) {
  const errors = [];
  const warnings = [];
  for (const adapter of registryAdaptersForNode(node)) {
    const valueSnapshot = preflight ? snapshotWidgetValues(node) : null;
    try {
      const result = (await adapter.refresh?.(node, target, { force, preflight })) ?? {};
      errors.push(...(Array.isArray(result.errors) ? result.errors : []));
      warnings.push(...(Array.isArray(result.warnings) ? result.warnings : []));
    } catch (error) {
      const message = `${adapter.id} failed for ${nodeClassType(node)}: ${error?.message ?? error}`;
      for (const inputName of adapterManagedInputs(adapter, node)) {
        applyRemoteWidgetOptions(node, widgetByName(node, inputName), target, [], { error: message });
      }
      errors.push(message);
    }
    const changedInputs = changedWidgetNames(node, valueSnapshot);
    if (changedInputs.length) {
      errors.push(
        `${adapter.id} updated widget values during remote preflight (${changedInputs.join(
          ", ",
        )}). Review the node and queue again so ComfyUI serializes the updated values.`,
      );
    }
  }
  return { errors, warnings };
}

function applyDefinitionToNode(node, target, definition) {
  const errors = [];
  const warnings = [];
  const classType = nodeClassType(node);
  node.__cutleryRemoteDefinition = definition ?? null;
  restoreRemoteUploadActions(node);
  if (!definition?.available) {
    const message =
      definition?.error ||
      definition?.errors?.[0]?.message ||
      `Node class ${classType} is not installed on ${target}.`;
    for (const inputName of declaredLocalComboInputNames(node)) {
      if (isRemoteModelNode(node) && inputName === "model_name") {
        continue;
      }
      applyRemoteWidgetOptions(node, widgetByName(node, inputName), target, [], { error: message });
    }
    disableRemoteUploadActions(node, target);
    errors.push(message);
    return { errors, warnings };
  }
  if (definition.compatible === false) {
    const message =
      definition?.errors?.[0]?.message ||
      `Node class ${classType} could not provide a compatible definition on ${target}.`;
    shieldNodeFromLocalChoices(node, target, message);
    errors.push(message);
    return { errors, warnings };
  }

  const remoteInputs = remoteComboInputs(definition);
  const allRemoteInputs = remoteInputRecords(definition);
  if (Object.values(remoteInputs).some((input) => input.upload_backed)) {
    disableRemoteUploadActions(node, target);
    warnings.push("Uploads are disabled for remote groups; choose a file that already exists on the remote target.");
  }
  const localComboNames = declaredLocalComboInputNames(node);
  const managedInputNames = registryManagedInputNames(node);
  const dynamicInputNames = Object.entries(allRemoteInputs)
    .filter(([, input]) => input?.kind === "dynamic")
    .map(([inputName]) => inputName);
  for (const inputName of new Set([...localComboNames, ...Object.keys(remoteInputs), ...dynamicInputNames])) {
    if (isRemoteModelNode(node) && inputName === "model_name") {
      continue;
    }
    const remoteInputRecord = allRemoteInputs[inputName];
    if (remoteInputRecord?.kind === "dynamic") {
      if (managedInputNames.has(inputName)) {
        continue;
      }
      const widget = widgetByName(node, inputName);
      if (!widget) {
        continue;
      }
      const owner = String(remoteInputRecord.registry ?? "").trim();
      const message = owner
        ? `${classType}.${inputName} requires remote registry adapter ${owner}, but that adapter is not loaded.`
        : `${classType}.${inputName} is a dynamic remote input with no concrete option adapter.`;
      applyRemoteWidgetOptions(node, widget, target, [], { error: message });
      errors.push(message);
      continue;
    }
    const widget = widgetByName(node, inputName);
    if (!widget) {
      continue;
    }
    const remoteInput = remoteInputs[inputName];
    if (!remoteInput) {
      const message = `${classType}.${inputName} is not a combo input on ${target}.`;
      applyRemoteWidgetOptions(node, widget, target, [], { error: message });
      errors.push(message);
      continue;
    }
    if (!Array.isArray(remoteInput.options)) {
      const message = `${classType}.${inputName} returned malformed remote options.`;
      applyRemoteWidgetOptions(node, widget, target, [], { error: message });
      errors.push(message);
      continue;
    }

    const choices = typedUnique(remoteInput.options);
    applyRemoteWidgetOptions(node, widget, target, choices, {
      uploadBacked: remoteInput.upload_backed,
    });
    if (!selectedValueIsAvailable(widget, choices)) {
      if (remoteInput.materializable) {
        warnings.push(`${classType}.${inputName} is absent remotely and will be materialized when queued.`);
      } else {
        const message = choices.length
          ? `${classType}.${inputName} value ${JSON.stringify(widget.value)} is not available on ${target}.`
          : `${classType}.${inputName} has no valid choices on ${target}.`;
        widget.__cutleryRemoteOptionsError = message;
        errors.push(message);
      }
    }
  }
  return { errors, warnings };
}

function shieldNodeFromLocalChoices(node, target, error) {
  delete node.__cutleryRemoteDefinition;
  const names = declaredLocalComboInputNames(node);
  for (const inputName of names) {
    if (isRemoteModelNode(node) && inputName === "model_name") {
      continue;
    }
    const widget = widgetByName(node, inputName);
    const currentState = widget ? widgetOverlayStates.get(widget) : null;
    if (widget && (!currentState || currentState.target !== target)) {
      applyRemoteWidgetOptions(node, widget, target, [], { error });
    } else if (widget) {
      widget.__cutleryRemoteOptionsError = error;
    }
  }
  disableRemoteUploadActions(node, target);
}

async function refreshRemoteModelNode(node, target, { force = false } = {}) {
  hideWidget(node, "remote_target");
  setRemoteTargetWidget(node, target);
  const widget = widgetByName(node, "model_name");
  if (!widget) {
    return { errors: [`${MODEL_NODE_CLASS}.model_name widget is missing.`], warnings: [] };
  }
  const modelType = String(widgetByName(node, "model_type")?.value ?? "checkpoints");
  try {
    const choices = await fetchModelChoices(target, modelType, { force });
    applyRemoteWidgetOptions(node, widget, target, choices);
    const warnings = selectedValueIsAvailable(widget, choices)
      ? []
      : [`${MODEL_NODE_CLASS}.model_name is absent remotely and may be copied when queued.`];
    return { errors: [], warnings };
  } catch (error) {
    const message = `Could not load ${modelType} choices from ${target}: ${error?.message ?? error}`;
    const state = widgetOverlayStates.get(widget);
    if (!state || state.target !== target) {
      applyRemoteWidgetOptions(node, widget, target, [], { error: message });
    } else {
      widget.__cutleryRemoteOptionsError = message;
    }
    return { errors: [message], warnings: [] };
  }
}

function graphNodes() {
  const graph = app.graph ?? app.canvas?.graph;
  const nodes = new Set();
  for (const context of graphContexts(graph)) {
    for (const node of context.graph?._nodes ?? []) {
      nodes.add(node);
    }
  }
  return [...nodes];
}

function restoreNodesOutsideRemoteGroups(inspection) {
  const nodes = new Set([...overlayNodes, ...graphNodes()]);
  for (const node of nodes) {
    if (!inspection.membershipByNode.has(node)) {
      if (!overlayNodes.has(node) && !node.__cutleryRemoteOptionsStatus) {
        continue;
      }
      restoreRemoteOptionsForNode(node);
      restoreRegistryAdapters(node);
      if (isRemoteModelNode(node)) {
        setRemoteTargetWidget(node, "");
      }
      clearNodeRemoteStatus(node);
    }
  }
}

async function refreshRemoteOptionsNow({ force = false, preflight = false, inspection: requestedInspection = null } = {}) {
  if (graphRefreshTimer != null) {
    window.clearTimeout(graphRefreshTimer);
    graphRefreshTimer = null;
  }
  const graph = app.graph ?? app.canvas?.graph;
  if (!graph) {
    return inspectRemoteGroupMembership(null);
  }
  const generation = ++refreshGeneration;
  const inspection = requestedInspection ?? inspectRemoteGroupMembership(graph);
  restoreNodesOutsideRemoteGroups(inspection);

  for (const { node, message } of inspection.errors) {
    restoreRemoteOptionsForNode(node);
    restoreRegistryAdapters(node);
    if (isRemoteModelNode(node)) {
      setRemoteTargetWidget(node, "");
    }
    setNodeRemoteStatus(node, null, [message]);
  }

  const groupsByTarget = new Map();
  for (const groupInfo of inspection.groups) {
    const nodes = groupsByTarget.get(groupInfo.target) ?? new Set();
    for (const node of groupInfo.insideNodes) {
      nodes.add(node);
    }
    groupsByTarget.set(groupInfo.target, nodes);
  }

  await Promise.all(
    [...groupsByTarget.entries()].map(async ([target, nodes]) => {
      const classTypes = [...nodes].map(nodeClassType).filter(Boolean);
      let definitions = null;
      let definitionError = null;
      try {
        definitions = await fetchRemoteNodeDefinitions(target, classTypes, { force });
      } catch (error) {
        definitionError = `Could not load remote node definitions from ${target}: ${error?.message ?? error}`;
      }
      if (generation !== refreshGeneration) {
        return;
      }

      await Promise.all(
        [...nodes].map(async (node) => {
          if (generation !== refreshGeneration || remoteTargetForNode(node) !== target) {
            return;
          }
          const errors = [];
          const warnings = [];
          if (definitionError) {
            shieldNodeFromLocalChoices(node, target, definitionError);
            errors.push(definitionError);
          } else {
            const result = applyDefinitionToNode(node, target, definitions?.[nodeClassType(node)]);
            errors.push(...result.errors);
            warnings.push(...result.warnings);
          }
          if (isRemoteModelNode(node)) {
            const result = await refreshRemoteModelNode(node, target, { force });
            if (generation !== refreshGeneration || remoteTargetForNode(node) !== target) {
              return;
            }
            errors.push(...result.errors);
            warnings.push(...result.warnings);
          }
          const registryResult = await refreshRegistryAdapters(node, target, { force, preflight });
          if (generation !== refreshGeneration || remoteTargetForNode(node) !== target) {
            return;
          }
          errors.push(...registryResult.errors);
          warnings.push(...registryResult.warnings);
          setNodeRemoteStatus(node, target, errors, warnings);
        }),
      );
    }),
  );
  return requestedInspection ?? inspectRemoteGroupMembership(graph);
}

function scheduleRemoteOptionsRefresh({ force = false, delay = GRAPH_REFRESH_DELAY_MS } = {}) {
  if (graphRefreshTimer != null) {
    window.clearTimeout(graphRefreshTimer);
  }
  graphRefreshTimer = window.setTimeout(() => {
    graphRefreshTimer = null;
    return refreshRemoteOptionsNow({ force }).catch((error) => {
      console.warn("[Cutlery Remote Models] Failed to refresh remote widget options", error);
    });
  }, delay);
}

function installSharedRemoteGroupApi() {
  const shared =
    globalThis.cutleryRemoteGroups && typeof globalThis.cutleryRemoteGroups === "object"
      ? globalThis.cutleryRemoteGroups
      : {};
  Object.assign(shared, {
    isNodeInRemoteGroup(node) {
      return Boolean(remoteTargetForNode(node));
    },
    nodeRemoteTarget(node) {
      return remoteTargetForNode(node);
    },
    scheduleRemoteWidgetRefresh(options = {}) {
      scheduleRemoteOptionsRefresh({
        force: options.force ?? true,
        delay: options.delay ?? 0,
      });
    },
    applyRemoteWidgetOptions(node, inputName, values, metadata = {}) {
      const target = metadata.target ?? remoteTargetForNode(node);
      if (!target || remoteTargetForNode(node) !== target) {
        return { applied: false, selectedAvailable: false };
      }
      const widget = widgetByName(node, inputName);
      if (!widget) {
        return { applied: false, selectedAvailable: false };
      }
      const choices = typedUnique(values);
      applyRemoteWidgetOptions(node, widget, target, choices, metadata);
      return {
        applied: true,
        selectedAvailable: selectedValueIsAvailable(widget, choices),
        choices,
      };
    },
    registerRegistryAdapter(adapter) {
      const id = String(adapter?.id ?? "").trim();
      if (!id || typeof adapter?.refresh !== "function") {
        throw new Error("Remote registry adapters require a stable id and refresh function.");
      }
      registryAdapters.set(id, { ...adapter, id });
      scheduleRemoteOptionsRefresh({ force: true, delay: 0 });
    },
  });
  globalThis.cutleryRemoteGroups = shared;
}

function ensureGraphRefreshInstalled() {
  if (graphRefreshInstalled || typeof api?.addEventListener !== "function") {
    return;
  }
  api.addEventListener("graphChanged", () => scheduleRemoteOptionsRefresh());
  graphRefreshInstalled = true;
}

function remoteModelNodesInGraph() {
  return graphNodes().filter(isRemoteModelNode);
}

function serializeRemoteTargetsForModelNodes(inspection = null) {
  const currentInspection = inspection ?? inspectRemoteGroupMembership(app.graph ?? app.canvas?.graph);
  for (const node of remoteModelNodesInGraph()) {
    setRemoteTargetWidget(node, remoteTargetForNode(node, currentInspection));
  }
}

function remapPartialExecutionTargets(body, remaps) {
  if (body.partial_execution_targets == null) {
    return;
  }
  if (!Array.isArray(body.partial_execution_targets)) {
    throw new Error("partial_execution_targets must be an array when Cutlery remote groups are present.");
  }
  const seen = new Set();
  body.partial_execution_targets = body.partial_execution_targets
    .map((target) => remaps.get(String(target)) ?? target)
    .filter((target) => {
      const key = String(target);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

async function preflightRemoteGroups(requestedInspection = null) {
  const inspection = await refreshRemoteOptionsNow({
    preflight: true,
    inspection: requestedInspection,
  });
  if (inspection.errors.length) {
    throw new Error(inspection.errors.map((record) => record.message).join("\n"));
  }
  const errors = [];
  for (const groupInfo of inspection.groups) {
    for (const node of groupInfo.insideNodes) {
      if (node.__cutleryRemoteOptionsError) {
        errors.push(
          `[${groupInfo.target}] ${nodeClassType(node) || "Unknown node"} ${nodeId(node) || ""}: ${node.__cutleryRemoteOptionsError}`,
        );
      }
    }
  }
  if (errors.length) {
    throw new Error(`Cutlery remote group preflight failed:\n${errors.join("\n")}`);
  }
  return inspection;
}

async function compileRemoteGroupPrompt(workflow, prompt, partialExecutionTargets, fetchApi = api.fetchApi.bind(api)) {
  const compileResponse = await fetchApi("/cutlery/remote/compile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workflow,
      prompt,
      partial_execution_targets: partialExecutionTargets,
    }),
  });
  const compilation = await compileResponse.json();
  if (!compileResponse.ok || compilation?.ok !== true || !compilation?.prompt) {
    throw new Error(compilation?.error || `Cutlery remote compilation failed with HTTP ${compileResponse.status}.`);
  }
  return compilation;
}

function downloadApiWorkflow(prompt) {
  const blob = new Blob([JSON.stringify(prompt, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "workflow_api.json";
  link.click();
  URL.revokeObjectURL(url);
}

async function exportRemoteAwareApiWorkflow() {
  const graph = app.graph ?? app.canvas?.graph;
  const initial = await app.graphToPrompt();
  const inspection = remoteGroupInspectionForPrompt(graph, initial.output);
  if (!inspection.groups.length && !inspection.errors.length) {
    downloadApiWorkflow(initial.output);
    return initial.output;
  }

  const refreshedInspection = await preflightRemoteGroups(inspection);
  serializeRemoteTargetsForModelNodes(refreshedInspection);
  const prepared = await app.graphToPrompt();
  const compilation = await compileRemoteGroupPrompt(prepared.workflow, prepared.output);
  downloadApiWorkflow(compilation.prompt);
  return compilation.prompt;
}

function requestPath(input) {
  const raw = typeof input === "string" ? input : input?.url;
  if (!raw) {
    return "";
  }
  try {
    return new URL(raw, globalThis.window?.location?.origin ?? "http://localhost").pathname;
  } catch (_error) {
    return String(raw).split("?")[0];
  }
}

function isPromptPost(input, options) {
  const method = String(options?.method ?? input?.method ?? "GET").toUpperCase();
  return method === "POST" && requestPath(input) === "/prompt";
}

function installPromptFetchHook() {
  if (fetchHookInstalled || typeof api?.fetchApi !== "function") {
    return;
  }
  const originalFetchApi = api.fetchApi.bind(api);
  api.fetchApi = async function cutleryRemoteModelsFetchApi(input, options = {}) {
    let nextOptions = options;
    if (isPromptPost(input, options) && typeof options?.body === "string") {
      try {
        const body = JSON.parse(options.body);
        const graph = app.graph ?? app.canvas?.graph;
        const promptInspection = remoteGroupInspectionForPrompt(graph, body?.prompt);
        if (promptInspection.groups.length || promptInspection.errors.length) {
          const inspection = await preflightRemoteGroups(promptInspection);
          serializeRemoteTargetsForModelNodes(inspection);
        }
        const editorWorkflow =
          body?.extra_data?.extra_pnginfo?.workflow ??
          (typeof graph?.serialize === "function" ? graph.serialize() : null);
        if (!editorWorkflow || typeof editorWorkflow !== "object") {
          if (promptInspection.groups.length || promptInspection.errors.length) {
            throw new Error("Cutlery remote groups need the serialized editor workflow for canonical compilation.");
          }
          return originalFetchApi(input, options);
        }
        const compilation = await compileRemoteGroupPrompt(
          editorWorkflow,
          body.prompt,
          body.partial_execution_targets,
          originalFetchApi,
        );
        const compiledBody = { ...body, prompt: compilation.prompt };
        remapPartialExecutionTargets(compiledBody, new Map(Object.entries(compilation.remaps ?? {})));
        nextOptions = {
          ...options,
          body: JSON.stringify(compiledBody),
        };
      } catch (error) {
        console.warn("[Cutlery Remote Models] Failed remote group preflight or compilation", error);
        throw error;
      }
    }
    return originalFetchApi(input, nextOptions);
  };
  fetchHookInstalled = true;
}

function installModelTypeCallback(node) {
  const widget = widgetByName(node, "model_type");
  if (!widget || widget.__cutleryRemoteModelTypeCallbackInstalled) {
    return;
  }
  const original = widget.callback;
  widget.callback = function remoteModelTypeCallback(value, ...args) {
    const result = original?.call(this, value, ...args);
    scheduleRemoteOptionsRefresh({ force: true, delay: 0 });
    return result;
  };
  widget.__cutleryRemoteModelTypeCallbackInstalled = true;
}

function installRemoteModelNode(nodeType) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function onCutleryRemoteModelCreated(...args) {
    const result = originalOnNodeCreated?.apply(this, args);
    hideWidget(this, "remote_target");
    installModelTypeCallback(this);
    scheduleRemoteOptionsRefresh({ delay: 0 });
    return result;
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function configureCutleryRemoteModel(...args) {
    const result = originalConfigure?.apply(this, args);
    hideWidget(this, "remote_target");
    installModelTypeCallback(this);
    scheduleRemoteOptionsRefresh({ delay: 0 });
    return result;
  };
}

function installRemoteGroupExecutorNode(nodeType) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function onCutleryRemoteGroupExecutorCreated(...args) {
    const result = originalOnNodeCreated?.apply(this, args);
    scheduleRefreshExecutorPorts(this);
    return result;
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function configureCutleryRemoteGroupExecutor(...args) {
    const result = originalConfigure?.apply(this, args);
    scheduleRefreshExecutorPorts(this);
    return result;
  };
}

function comboInputNamesFromNodeData(nodeData) {
  const names = new Set();
  for (const sectionName of ["required", "optional", "hidden"]) {
    const section = nodeData?.input?.[sectionName];
    if (!section || Array.isArray(section) || typeof section !== "object") {
      continue;
    }
    for (const [inputName, spec] of Object.entries(section)) {
      if (!Array.isArray(spec)) {
        continue;
      }
      if (Array.isArray(spec[0]) || (spec[0] === "COMBO" && Array.isArray(spec[1]?.options))) {
        names.add(inputName);
      }
    }
  }
  return names;
}

installSharedRemoteGroupApi();

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: "Comfy.ExportWorkflowAPI",
      icon: "pi pi-download",
      label: "Export Workflow (API Format)",
      menubarLabel: "Export (API)",
      function: exportRemoteAwareApiWorkflow,
    },
  ],
  setup() {
    installSharedRemoteGroupApi();
    ensureGraphRefreshInstalled();
    installPromptFetchHook();
    scheduleRemoteOptionsRefresh({ delay: 0 });
  },
  nodeCreated(node) {
    ensureGraphRefreshInstalled();
    installPromptFetchHook();
    ensureRemoteStatusRenderer(node);
    if (isRemoteModelNode(node)) {
      hideWidget(node, "remote_target");
      installModelTypeCallback(node);
    }
    scheduleRemoteOptionsRefresh({ delay: 0 });
  },
  loadedGraphNode(node) {
    ensureRemoteStatusRenderer(node);
    if (isRemoteModelNode(node)) {
      hideWidget(node, "remote_target");
      installModelTypeCallback(node);
    }
    scheduleRemoteOptionsRefresh({ delay: 0 });
  },
  afterConfigureGraph() {
    scheduleRemoteOptionsRefresh({ delay: 0 });
  },
  async refreshComboInNodes() {
    await refreshRemoteOptionsNow({ force: true });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    nodeType.__cutleryClassName = nodeData?.name ?? nodeType.__cutleryClassName;
    nodeType.prototype.__cutleryClassName = nodeData?.name ?? nodeType.prototype.__cutleryClassName;
    nodeType.prototype.__cutleryLocalComboInputNames = comboInputNamesFromNodeData(nodeData);
    if (nodeData?.name === MODEL_NODE_CLASS) {
      nodeType.prototype.comfyClass = nodeData.name;
      if (!nodeType.prototype.__cutleryRemoteModelsInstalled) {
        installRemoteModelNode(nodeType);
        nodeType.prototype.__cutleryRemoteModelsInstalled = true;
      }
    }
    if (GROUP_EXECUTOR_CLASSES.has(nodeData?.name)) {
      nodeType.prototype.comfyClass = nodeData.name;
      stripInitialExecutorPorts(nodeData);
      if (!nodeType.prototype.__cutleryRemoteGroupExecutorInstalled) {
        installRemoteGroupExecutorNode(nodeType);
        nodeType.prototype.__cutleryRemoteGroupExecutorInstalled = true;
      }
    }
  },
});
