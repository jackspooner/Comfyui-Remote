from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class ProgressMappingError(ValueError):
    """Raised when compiled peer-to-local progress metadata is not trustworthy."""


class ProgressEventError(ValueError):
    """Raised when a remote progress event cannot safely be mirrored."""


@dataclass(frozen=True)
class ProgressNodeIdentity:
    """Immutable local identity for one peer API node."""

    api_node_id: str
    display_node_id: str
    parent_node_id: str
    real_node_id: str
    subgraph_instance: str
    group: str
    target: str
    visible: bool


@dataclass(frozen=True)
class RemoteProgressEvent:
    prompt_id: str
    node_id: str
    value: float
    max_value: float
    sequence: int | None = None


@dataclass(frozen=True)
class MirroredProgressUpdate:
    """A ComfyUI-adaptable update for one original API node."""

    prompt_id: str
    node_id: str
    value: float
    max_value: float
    first: bool
    terminal: bool

    def as_progress_data(self) -> dict[str, float | str]:
        """Return the plain payload shape used by a typical progress transport."""

        return {
            "prompt_id": self.prompt_id,
            "node": self.node_id,
            "value": self.value,
            "max": self.max_value,
        }


ProgressEmitter = Callable[[MirroredProgressUpdate], None]
Clock = Callable[[], float]

_IDENTITY_FIELDS = frozenset(
    {
        "api_node_id",
        "display_node_id",
        "parent_node_id",
        "real_node_id",
        "subgraph_instance",
        "group",
        "target",
        "visible",
    }
)


def parse_progress_mapping(payload: object) -> Mapping[str, ProgressNodeIdentity]:
    """Validate compiled peer-node metadata and expose an immutable mapping.

    The compiler may wrap its peer-node map in ``{"nodes": ...}``; no remote
    prompt id is encoded here because each execution receives one at runtime.
    """

    source = _mapping(payload, "progress mapping")
    if "nodes" in source:
        if set(source) != {"nodes"}:
            raise ProgressMappingError("A wrapped progress mapping may contain only 'nodes'.")
        source = _mapping(source["nodes"], "progress mapping nodes")

    entries: dict[str, ProgressNodeIdentity] = {}
    for peer_node_id, raw_identity in source.items():
        peer_id = _non_empty_string(peer_node_id, "peer node id", ProgressMappingError)
        if peer_id in entries:
            raise ProgressMappingError(f"Duplicate peer node id {peer_id!r}.")
        identity = _mapping(raw_identity, f"progress identity for peer node {peer_id!r}")
        unknown_fields = set(identity) - _IDENTITY_FIELDS
        missing_fields = _IDENTITY_FIELDS - set(identity)
        if unknown_fields or missing_fields:
            detail = []
            if missing_fields:
                detail.append(f"missing {', '.join(sorted(missing_fields))}")
            if unknown_fields:
                detail.append(f"unknown {', '.join(sorted(unknown_fields))}")
            raise ProgressMappingError(f"Progress identity for peer node {peer_id!r} has {'; '.join(detail)} fields.")
        visible = identity["visible"]
        if type(visible) is not bool:
            raise ProgressMappingError(f"Progress identity for peer node {peer_id!r} has a non-boolean visible flag.")
        api_node_id = _string(identity["api_node_id"], "api_node_id", ProgressMappingError)
        display_node_id = _string(identity["display_node_id"], "display_node_id", ProgressMappingError)
        if visible and not api_node_id:
            raise ProgressMappingError(f"Progress identity for peer node {peer_id!r} must have an api_node_id when visible.")
        if visible and not display_node_id:
            raise ProgressMappingError(f"Progress identity for peer node {peer_id!r} must have a display_node_id when visible.")
        entries[peer_id] = ProgressNodeIdentity(
            api_node_id=api_node_id,
            display_node_id=display_node_id,
            parent_node_id=_string(identity["parent_node_id"], "parent_node_id", ProgressMappingError),
            real_node_id=_string(identity["real_node_id"], "real_node_id", ProgressMappingError),
            subgraph_instance=_string(identity["subgraph_instance"], "subgraph_instance", ProgressMappingError),
            group=_non_empty_string(identity["group"], "group", ProgressMappingError),
            target=_non_empty_string(identity["target"], "target", ProgressMappingError),
            visible=visible,
        )
    return MappingProxyType(entries)


class ProgressMirror:
    """Aggregate validated peer progress into original local API-node updates.

    This class intentionally knows nothing about ComfyUI's progress registry or
    websocket server. The caller supplies an emitter that translates the
    immutable update into either integration surface.
    """

    def __init__(
        self,
        *,
        local_prompt_id: str,
        remote_prompt_id: str,
        mapping: Mapping[str, ProgressNodeIdentity] | object,
        emitter: ProgressEmitter,
        clock: Clock = time.monotonic,
        updates_per_second: float = 10.0,
    ):
        self.local_prompt_id = _non_empty_string(local_prompt_id, "local_prompt_id", ProgressEventError)
        self.remote_prompt_id = _non_empty_string(remote_prompt_id, "remote_prompt_id", ProgressEventError)
        if _is_identity_mapping(mapping):
            self.mapping = MappingProxyType(dict(mapping))
        else:
            self.mapping = parse_progress_mapping(mapping)
        if not callable(emitter):
            raise TypeError("emitter must be callable.")
        if not isinstance(updates_per_second, (int, float)) or isinstance(updates_per_second, bool) or updates_per_second <= 0:
            raise ValueError("updates_per_second must be positive.")
        self._emitter = emitter
        self._clock = clock
        self._minimum_interval = 1.0 / float(updates_per_second)
        self._remote: dict[str, RemoteProgressEvent] = {}
        self._local: dict[str, tuple[float, float]] = {}
        self._last_sequence: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}
        self._pending: set[str] = set()
        self._emitted: set[str] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def ingest(self, payload: RemoteProgressEvent | Mapping[str, Any]) -> MirroredProgressUpdate | None:
        """Validate and mirror one remote event, rejecting stale or foreign events."""

        self._require_open()
        event = payload if isinstance(payload, RemoteProgressEvent) else parse_remote_progress_event(payload)
        if event.prompt_id != self.remote_prompt_id:
            raise ProgressEventError(f"Remote progress event belongs to unknown prompt {event.prompt_id!r}.")
        identity = self.mapping.get(event.node_id)
        if identity is None:
            raise ProgressEventError(f"Remote progress event references unknown peer node {event.node_id!r}.")
        if not identity.visible:
            return None
        previous = self._remote.get(event.node_id)
        self._validate_order(event, previous)
        self._remote[event.node_id] = event
        if event.sequence is not None:
            self._last_sequence[event.node_id] = event.sequence
        return self._consider_emit(identity.api_node_id)

    def set_local_progress(self, api_node_id: str, value: object, max_value: object) -> MirroredProgressUpdate | None:
        """Add optional local work to the same aggregate as duplicated peer work."""

        self._require_open()
        api_id = _non_empty_string(api_node_id, "api_node_id", ProgressEventError)
        if api_id not in {identity.api_node_id for identity in self.mapping.values()}:
            raise ProgressEventError(f"Local progress references unknown original API node {api_id!r}.")
        normalized_value, normalized_max = _progress_values(value, max_value, ProgressEventError)
        previous = self._local.get(api_id)
        if previous is not None and (normalized_value < previous[0] or normalized_max < previous[1]):
            raise ProgressEventError(f"Local progress for API node {api_id!r} is out of order.")
        self._local[api_id] = (normalized_value, normalized_max)
        return self._consider_emit(api_id)

    def flush(self, *, force: bool = False) -> tuple[MirroredProgressUpdate, ...]:
        """Emit due coalesced updates; callers may schedule this with their UI loop."""

        self._require_open()
        updates: list[MirroredProgressUpdate] = []
        for api_node_id in tuple(sorted(self._pending)):
            if force or self._is_due(api_node_id):
                update = self._emit(api_node_id)
                if update is not None:
                    updates.append(update)
        return tuple(updates)

    def succeed(self) -> None:
        self.clear()

    def fail(self) -> None:
        self.clear()

    def cancel(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._remote.clear()
        self._local.clear()
        self._last_sequence.clear()
        self._last_emit.clear()
        self._pending.clear()
        self._emitted.clear()
        self._closed = True

    def _validate_order(self, event: RemoteProgressEvent, previous: RemoteProgressEvent | None) -> None:
        previous_sequence = self._last_sequence.get(event.node_id)
        if event.sequence is not None and previous_sequence is not None and event.sequence <= previous_sequence:
            raise ProgressEventError(f"Remote progress event for peer node {event.node_id!r} is out of order.")
        if previous is not None and (event.value < previous.value or event.max_value < previous.max_value):
            raise ProgressEventError(f"Remote progress event for peer node {event.node_id!r} is out of order.")

    def _consider_emit(self, api_node_id: str) -> MirroredProgressUpdate | None:
        value, max_value = self._aggregate(api_node_id)
        first = api_node_id not in self._emitted
        terminal = value == max_value
        if first or terminal or self._is_due(api_node_id):
            return self._emit(api_node_id, first=first, terminal=terminal)
        self._pending.add(api_node_id)
        return None

    def _aggregate(self, api_node_id: str) -> tuple[float, float]:
        value = 0.0
        max_value = 0.0
        for peer_node_id, event in self._remote.items():
            if self.mapping[peer_node_id].api_node_id == api_node_id:
                value += event.value
                max_value += event.max_value
        local = self._local.get(api_node_id)
        if local is not None:
            value += local[0]
            max_value += local[1]
        return value, max_value

    def _emit(self, api_node_id: str, *, first: bool | None = None, terminal: bool | None = None) -> MirroredProgressUpdate | None:
        value, max_value = self._aggregate(api_node_id)
        if max_value <= 0:
            return None
        update = MirroredProgressUpdate(
            prompt_id=self.local_prompt_id,
            node_id=api_node_id,
            value=value,
            max_value=max_value,
            first=api_node_id not in self._emitted if first is None else first,
            terminal=value == max_value if terminal is None else terminal,
        )
        self._emitter(update)
        self._last_emit[api_node_id] = self._clock()
        self._emitted.add(api_node_id)
        self._pending.discard(api_node_id)
        return update

    def _is_due(self, api_node_id: str) -> bool:
        last_emit = self._last_emit.get(api_node_id)
        return last_emit is None or self._clock() - last_emit >= self._minimum_interval

    def _require_open(self) -> None:
        if self._closed:
            raise ProgressEventError("Progress mirror has been cleared.")


def parse_remote_progress_event(payload: Mapping[str, Any] | object) -> RemoteProgressEvent:
    source = _mapping(payload, "remote progress event", ProgressEventError)
    if "data" in source:
        source = _mapping(source["data"], "remote progress event data", ProgressEventError)
    node_id = source.get("node_id", source.get("node"))
    if "node_id" in source and "node" in source and source["node_id"] != source["node"]:
        raise ProgressEventError("Remote progress event has conflicting node and node_id values.")
    value, max_value = _progress_values(source.get("value"), source.get("max"), ProgressEventError)
    sequence = source.get("sequence")
    if sequence is not None and (type(sequence) is not int or sequence < 0):
        raise ProgressEventError("Remote progress event sequence must be a non-negative integer.")
    return RemoteProgressEvent(
        prompt_id=_non_empty_string(source.get("prompt_id"), "prompt_id", ProgressEventError),
        node_id=_non_empty_string(node_id, "node_id", ProgressEventError),
        value=value,
        max_value=max_value,
        sequence=sequence,
    )


def _is_identity_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(isinstance(identity, ProgressNodeIdentity) for identity in value.values())


def _mapping(value: object, name: str, error_type: type[ValueError] = ProgressMappingError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{name} must be an object.")
    return value


def _non_empty_string(value: object, name: str, error_type: type[ValueError]) -> str:
    text = _string(value, name, error_type)
    if not text:
        raise error_type(f"{name} must be a non-empty string.")
    return text


def _string(value: object, name: str, error_type: type[ValueError]) -> str:
    if not isinstance(value, str):
        raise error_type(f"{name} must be a string.")
    return value


def _progress_values(value: object, max_value: object, error_type: type[ValueError]) -> tuple[float, float]:
    if not _is_number(value) or not _is_number(max_value):
        raise error_type("Remote progress value and max must be finite numbers.")
    normalized_value = float(value)
    normalized_max = float(max_value)
    if not math.isfinite(normalized_value) or not math.isfinite(normalized_max) or normalized_max <= 0:
        raise error_type("Remote progress max must be a positive finite number.")
    if normalized_value < 0 or normalized_value > normalized_max:
        raise error_type("Remote progress value must be between zero and max.")
    return normalized_value, normalized_max


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
