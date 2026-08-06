import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "CutleryRemoteClipTextEncode";
const DUAL_NODE_CLASS = "CutleryRemoteDualClipTextEncode";
const QWEN_IMAGE_EDIT_PLUS_NODE_CLASS = "CutleryRemoteTextEncodeQwenImageEditPlus";
const CHOICES_ENDPOINT = "/cutlery/remote/clip/choices";
const REMOTE_REGISTRY_ENDPOINT = "/cutlery/remote/proxy/registry";
const REMOTE_REGISTRY_ID = "remote_clip.choices";
const EXTENSION_NAME = "Cutlery.RemoteClipTextEncode";
const LEGACY_MANUAL_LORA_INPUT = /^(?:lora_name|lora|strength_clip)_[1-8]$/;
const CLIP_WIDGETS = ["text_encoder", "clip_name1", "clip_name2"];
const VAE_WIDGETS = ["vae_name"];
const NONE_CHOICE = "None";
const REMOTE_MANAGED_INPUTS = {
  [NODE_CLASS]: ["text_encoder", "clip_type"],
  [DUAL_NODE_CLASS]: ["clip_name1", "clip_name2", "clip_type"],
  [QWEN_IMAGE_EDIT_PLUS_NODE_CLASS]: ["text_encoder", "vae_name"],
};

let choicesPromise = null;
const remoteChoicesPromises = new Map();
let keyRefreshInstalled = false;

function uniqueStrings(values) {
  return [...new Set((values ?? []).map((value) => String(value ?? "").trim()).filter(Boolean))];
}

function isRemoteClipNode(node) {
  return (
    node?.comfyClass === NODE_CLASS ||
    node?.comfyClass === DUAL_NODE_CLASS ||
    node?.comfyClass === QWEN_IMAGE_EDIT_PLUS_NODE_CLASS ||
    node?.constructor?.comfyClass === NODE_CLASS ||
    node?.constructor?.comfyClass === DUAL_NODE_CLASS ||
    node?.constructor?.comfyClass === QWEN_IMAGE_EDIT_PLUS_NODE_CLASS ||
    node?.type === NODE_CLASS ||
    node?.type === DUAL_NODE_CLASS ||
    node?.type === QWEN_IMAGE_EDIT_PLUS_NODE_CLASS
  );
}

function widgetByName(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

function remoteTarget(node) {
  return globalThis.cutleryRemoteGroups?.nodeRemoteTarget?.(node) ?? null;
}

function setWidgetChoices(node, name, values) {
  const widget = widgetByName(node, name);
  const choices = uniqueStrings(values);
  if (!widget) {
    return { applied: false, selectedAvailable: false };
  }
  const target = remoteTarget(node);
  if (target) {
    const remoteGroups = globalThis.cutleryRemoteGroups;
    if (typeof remoteGroups?.applyRemoteWidgetOptions === "function") {
      return remoteGroups.applyRemoteWidgetOptions(node, name, choices, { target });
    }
    remoteGroups?.scheduleRemoteWidgetRefresh?.({ force: true, delay: 0 });
    return { applied: false, selectedAvailable: false };
  }
  if (!choices.length) {
    return { applied: false, selectedAvailable: false };
  }

  widget.type = "combo";
  widget.options = widget.options ?? {};
  widget.options.values = choices;
  if (!choices.includes(String(widget.value ?? ""))) {
    widget.value = choices[0];
  }
  return { applied: true, selectedAvailable: true, choices };
}

function promoteWidgetToCombo(node, name) {
  if (remoteTarget(node)) {
    globalThis.cutleryRemoteGroups?.scheduleRemoteWidgetRefresh?.({ force: true, delay: 0 });
    return;
  }
  const widget = widgetByName(node, name);
  if (!widget) {
    return;
  }
  const existingChoices = Array.isArray(widget.options?.values) ? widget.options.values : [];
  const choices = uniqueStrings([...existingChoices, widget.value]);
  if (!choices.length) {
    return;
  }

  widget.type = "combo";
  widget.options = widget.options ?? {};
  widget.options.values = choices;
}

function promoteRemoteModelWidgets(node) {
  if (!isRemoteClipNode(node)) {
    return;
  }
  for (const name of CLIP_WIDGETS) {
    promoteWidgetToCombo(node, name);
  }
}

async function fetchRemoteClipChoices(node = null) {
  const target = remoteTarget(node);
  if (target) {
    let request = remoteChoicesPromises.get(target);
    if (!request) {
      request = api
        .fetchApi(REMOTE_REGISTRY_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
          body: JSON.stringify({ target, registry: REMOTE_REGISTRY_ID, payload: {} }),
        })
        .then(async (response) => {
          const wrapper = await response.json().catch(() => ({}));
          if (!response.ok || wrapper?.ok === false) {
            throw new Error(wrapper.error || wrapper.message || `${response.status} ${response.statusText}`);
          }
          const payload = wrapper.payload;
          if (!payload || typeof payload !== "object" || Array.isArray(payload) || payload.ok === false) {
            throw new Error(payload?.error || `Remote registry ${REMOTE_REGISTRY_ID} returned an invalid payload.`);
          }
          return payload;
        })
        .finally(() => {
          remoteChoicesPromises.delete(target);
        });
      remoteChoicesPromises.set(target, request);
    }
    return request;
  }
  if (!choicesPromise) {
    choicesPromise = api
      .fetchApi(CHOICES_ENDPOINT, { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload?.ok === false) {
          throw new Error(payload.error || payload.message || `${response.status} ${response.statusText}`);
        }
        return payload;
      })
      .finally(() => {
        choicesPromise = null;
      });
  }
  return choicesPromise;
}

async function refreshRemoteClipChoices(node, { throwOnError = false } = {}) {
  if (!isRemoteClipNode(node)) {
    return {};
  }
  try {
    const requestedTarget = remoteTarget(node);
    const payload = await fetchRemoteClipChoices(node);
    if (requestedTarget && remoteTarget(node) !== requestedTarget) {
      throw new Error(`Discarded stale ${REMOTE_REGISTRY_ID} response for ${requestedTarget}.`);
    }
    for (const name of CLIP_WIDGETS) {
      setWidgetChoices(node, name, payload.text_encoders);
    }
    for (const name of VAE_WIDGETS) {
      setWidgetChoices(node, name, [NONE_CHOICE, ...(payload.vaes ?? [])]);
    }
    setWidgetChoices(node, "clip_type", payload.clip_types);
    return payload;
  } catch (error) {
    console.warn("[Cutlery Remote CLIP] Failed to refresh remote clip choices", error);
    if (throwOnError) {
      throw error;
    }
    return {};
  }
}

function scheduleRefreshRemoteClipChoices(node) {
  window.setTimeout(() => refreshRemoteClipChoices(node), 0);
}

function remoteClipNodesInGraph() {
  const graph = app.graph ?? app.canvas?.graph;
  return Array.from(graph?._nodes ?? []).filter(isRemoteClipNode);
}

function scheduleRefreshAllRemoteClipChoices() {
  window.setTimeout(() => Promise.all(remoteClipNodesInGraph().map((node) => refreshRemoteClipChoices(node))), 0);
}

function isTextEditingTarget(target) {
  if (!target) {
    return false;
  }
  const tagName = String(target.tagName ?? "").toLowerCase();
  return target.isContentEditable || tagName === "input" || tagName === "textarea" || tagName === "select";
}

function ensureRemoteClipKeyRefreshInstalled() {
  if (keyRefreshInstalled || typeof window?.addEventListener !== "function") {
    return;
  }
  window.addEventListener("keyup", (event) => {
    if (String(event?.key ?? "").toLowerCase() !== "r" || event?.ctrlKey || event?.altKey || event?.metaKey) {
      return;
    }
    if (isTextEditingTarget(event?.target ?? globalThis.document?.activeElement)) {
      return;
    }
    scheduleRefreshAllRemoteClipChoices();
  });
  keyRefreshInstalled = true;
}

function isLegacyManualLoraInput(input) {
  return LEGACY_MANUAL_LORA_INPUT.test(String(input?.name ?? ""));
}

function removeInputAt(node, index) {
  if (typeof node?.removeInput === "function") {
    node.removeInput(index);
    return;
  }
  node?.inputs?.splice?.(index, 1);
}

function resizeAfterInputCleanup(node) {
  const size = node?.computeSize?.();
  if (Array.isArray(size) && typeof node?.setSize === "function") {
    node.setSize(size);
  }
  const graph = node?.graph ?? app.graph ?? app.canvas?.graph;
  graph?.setDirtyCanvas?.(true, true);
}

function removeLegacyManualLoraInputs(node) {
  let changed = false;
  for (let index = (node?.inputs?.length ?? 0) - 1; index >= 0; index -= 1) {
    if (isLegacyManualLoraInput(node.inputs[index])) {
      removeInputAt(node, index);
      changed = true;
    }
  }

  if (Array.isArray(node?.widgets)) {
    const widgets = node.widgets.filter((widget) => !isLegacyManualLoraInput(widget));
    if (widgets.length !== node.widgets.length) {
      node.widgets = widgets;
      changed = true;
    }
  }

  if (changed) {
    resizeAfterInputCleanup(node);
  }
  return changed;
}

function remoteClipManagedInputs(node) {
  const classType = node?.comfyClass ?? node?.constructor?.comfyClass ?? node?.type;
  return REMOTE_MANAGED_INPUTS[classType] ?? [];
}

function remoteChoiceErrors(node, target) {
  const errors = [];
  for (const inputName of remoteClipManagedInputs(node)) {
    const widget = widgetByName(node, inputName);
    const choices = Array.isArray(widget?.options?.values) ? widget.options.values : [];
    if (!widget || !choices.length) {
      errors.push(`${node?.comfyClass ?? node?.type}.${inputName} has no valid choices on ${target}.`);
      continue;
    }
    if (!choices.includes(widget.value)) {
      errors.push(
        `${node?.comfyClass ?? node?.type}.${inputName} value ${JSON.stringify(widget.value)} is not available on ${target}.`,
      );
    }
  }
  return errors;
}

async function refreshRemoteClipRegistry(node, target) {
  if (remoteTarget(node) !== target) {
    return {};
  }
  await refreshRemoteClipChoices(node, { throwOnError: true });
  return { errors: remoteChoiceErrors(node, target) };
}

function restoreLocalRemoteClipRegistry(node) {
  scheduleRefreshRemoteClipChoices(node);
}

let remoteRegistryAdapterRegistered = false;
function registerRemoteRegistryAdapter() {
  if (remoteRegistryAdapterRegistered) {
    return;
  }
  const remoteGroups = globalThis.cutleryRemoteGroups;
  if (typeof remoteGroups?.registerRegistryAdapter !== "function") {
    return;
  }
  remoteGroups.registerRegistryAdapter({
    id: "cutlery.remote_clip.v1",
    classTypes: Object.keys(REMOTE_MANAGED_INPUTS),
    managedInputs: remoteClipManagedInputs,
    refresh: refreshRemoteClipRegistry,
    restore: restoreLocalRemoteClipRegistry,
  });
  remoteRegistryAdapterRegistered = true;
}

registerRemoteRegistryAdapter();

app.registerExtension({
  name: EXTENSION_NAME,
  setup() {
    registerRemoteRegistryAdapter();
  },
  nodeCreated(node) {
    registerRemoteRegistryAdapter();
    ensureRemoteClipKeyRefreshInstalled();
    promoteRemoteModelWidgets(node);
    scheduleRefreshRemoteClipChoices(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    registerRemoteRegistryAdapter();
    if (nodeData?.name !== NODE_CLASS && nodeData?.name !== DUAL_NODE_CLASS && nodeData?.name !== QWEN_IMAGE_EDIT_PLUS_NODE_CLASS) {
      return;
    }
    ensureRemoteClipKeyRefreshInstalled();
    nodeType.prototype.comfyClass = nodeData.name;
    if (nodeType.prototype.__cutleryRemoteClipLegacyInputCleanupInstalled) {
      return;
    }
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreated(...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      registerRemoteRegistryAdapter();
      removeLegacyManualLoraInputs(this);
      promoteRemoteModelWidgets(this);
      scheduleRefreshRemoteClipChoices(this);
      return result;
    };
    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function configure(...args) {
      const result = originalConfigure?.apply(this, args);
      registerRemoteRegistryAdapter();
      removeLegacyManualLoraInputs(this);
      promoteRemoteModelWidgets(this);
      scheduleRefreshRemoteClipChoices(this);
      return result;
    };
    nodeType.prototype.__cutleryRemoteClipLegacyInputCleanupInstalled = true;
  },
});
