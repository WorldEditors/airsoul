"""Helpers for converting benchmark trajectories into RoboFM V1 datasets.

The environment-specific generators are deliberately kept responsible for
sampling trajectories.  This module owns the storage contract: every output
record is a committed V1 dataset and never a collection of ``.npy`` files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import AtomType
from .writer import UnifiedDatasetWriter


BYTE_TOKEN_BASE = 16

DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<step>": 3,
    "<field>": 4,
    "</field>": 5,
    "<value>": 6,
    "</value>": 7,
    "<|message|>": 272,
    "<|system|>": 273,
    "<|user|>": 274,
    "<|assistant|>": 275,
    "<|tool|>": 276,
    "<|content|>": 277,
    "<|name|>": 278,
    "<|tool_call_id|>": 279,
    "<|tool_call|>": 280,
    "<|tool_result|>": 281,
    "<|eom|>": 282,
}

TOKENIZER_VOCAB_SIZE = max(DEFAULT_SPECIAL_TOKENS.values()) + 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _value_tokens(value: Any) -> list[int]:
    """Encode a scalar or non-image value as deterministic byte tokens."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            separators=(",", ":"),
        ).encode("ascii")
        raw = header + b"\0" + memoryview(array).cast("B").tobytes()
    else:
        encoded = _jsonable(value)
        raw = json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return [BYTE_TOKEN_BASE + byte for byte in raw]


def _is_image_field(name: str, value: Any) -> bool:
    if not isinstance(value, np.ndarray) or value.ndim < 2:
        return False
    lowered = name.lower()
    if re.search(r"image|frame|render", lowered):
        return True
    # Observation/state vectors are common in RL records; only treat them as
    # images when they have at least height/width/channel dimensions.
    return bool(re.search(r"bev|^obs$|observation", lowered)) and value.ndim >= 3


_ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
}


def trajectory_messages(fields: Mapping[str, Any], *, role: str = "tool",
                        name: str = "trajectory") -> Iterable[Mapping[str, Any]]:
    """Turn producer fields into standard tool messages, one message per step."""
    if role not in _ROLE_TOKENS:
        raise ValueError(f"unsupported message role: {role}")
    lengths = [len(value) for value in fields.values()
               if isinstance(value, np.ndarray) and value.ndim > 0]
    steps = max(lengths, default=1)
    for step in range(steps):
        content = []
        for field_name, raw_value in fields.items():
            value = raw_value
            if isinstance(raw_value, np.ndarray) and raw_value.ndim > 0:
                if step >= len(raw_value):
                    continue
                value = raw_value[step]
            elif step > 0:
                continue
            content.append({
                "type": "field",
                "name": str(field_name),
                "value": value,
                "image": _is_image_field(str(field_name), raw_value),
            })
        message = {
            "role": role,
            "name": name,
            "tool_call_id": f"trajectory-step-{step}",
            "content": content,
        }
        yield message


def _append_tokens(values: list[int], types: list[int], tokens: Sequence[int]) -> None:
    values.extend(tokens)
    types.extend([int(AtomType.LANGUAGE_TOKEN)] * len(tokens))


def _append_json(values: list[int], types: list[int], value: Any) -> None:
    _append_tokens(values, types, _value_tokens(value))


def _append_message_part(values: list[int], types: list[int], images: list[np.ndarray],
                         part: Any) -> None:
    if isinstance(part, str):
        _append_json(values, types, part)
        return
    if not isinstance(part, Mapping):
        raise TypeError("message content parts must be strings or mappings")
    part_type = str(part.get("type", "text"))
    if part_type in {"text", "input_text"}:
        _append_json(values, types, part.get("text", ""))
    elif part_type in {"image", "input_image", "image_url"}:
        image = part.get("image")
        if image is None:
            image = part.get("input_image")
        if image is None and isinstance(part.get("image_url"), np.ndarray):
            image = part["image_url"]
        if image is None:
            # A URL is still useful as a language input when raw pixels are unavailable.
            _append_json(values, types, part.get("image_url", ""))
        else:
            images.append(np.ascontiguousarray(np.asarray(image)))
            values.append(len(images) - 1)
            types.append(int(AtomType.IMAGE))
    elif part_type == "field":
        field_name = str(part.get("name", "field"))
        values.append(DEFAULT_SPECIAL_TOKENS["<field>"])
        types.append(int(AtomType.LANGUAGE_TOKEN))
        _append_json(values, types, field_name)
        values.append(DEFAULT_SPECIAL_TOKENS["<value>"])
        types.append(int(AtomType.LANGUAGE_TOKEN))
        if bool(part.get("image", False)):
            images.append(np.ascontiguousarray(np.asarray(part.get("value"))))
            values.append(len(images) - 1)
            types.append(int(AtomType.IMAGE))
        else:
            _append_json(values, types, part.get("value"))
        values.extend([DEFAULT_SPECIAL_TOKENS["</value>"], DEFAULT_SPECIAL_TOKENS["</field>"]])
        types.extend([int(AtomType.LANGUAGE_TOKEN)] * 2)
    else:
        raise ValueError(f"unsupported message content type: {part_type}")


def _append_tool_call(values: list[int], types: list[int], call: Mapping[str, Any]) -> None:
    values.append(DEFAULT_SPECIAL_TOKENS["<|tool_call|>"])
    types.append(int(AtomType.LANGUAGE_TOKEN))
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise TypeError("function call must contain a mapping")
    payload = {
        "id": call.get("id"),
        "name": function.get("name"),
        "arguments": function.get("arguments", {}),
    }
    _append_json(values, types, payload)


def write_unified_messages(
    output_path: str | Path,
    messages: Sequence[Mapping[str, Any]],
    *,
    producer: Mapping[str, Any] | None = None,
    max_atoms_per_shard: int = 10_000_000,
) -> Path:
    """Write OpenAI-compatible text/image/tool messages as a V1 dataset."""
    values: list[int] = [DEFAULT_SPECIAL_TOKENS["<bos>"]]
    types: list[int] = [int(AtomType.LANGUAGE_TOKEN)]
    images: list[np.ndarray] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("messages must contain mappings")
        role = str(message.get("role", ""))
        role_token = _ROLE_TOKENS.get(role)
        if role_token is None:
            raise ValueError(f"unsupported message role: {role}")
        values.extend([DEFAULT_SPECIAL_TOKENS["<|message|>"], DEFAULT_SPECIAL_TOKENS[role_token]])
        types.extend([int(AtomType.LANGUAGE_TOKEN)] * 2)
        if message.get("name") is not None:
            values.append(DEFAULT_SPECIAL_TOKENS["<|name|>"])
            types.append(int(AtomType.LANGUAGE_TOKEN))
            _append_json(values, types, message["name"])
        if message.get("tool_call_id") is not None:
            values.append(DEFAULT_SPECIAL_TOKENS["<|tool_call_id|>"])
            types.append(int(AtomType.LANGUAGE_TOKEN))
            _append_json(values, types, message["tool_call_id"])
        content = message.get("content")
        if content is not None:
            values.append(DEFAULT_SPECIAL_TOKENS["<|content|>"])
            types.append(int(AtomType.LANGUAGE_TOKEN))
            parts = [content] if isinstance(content, str) else content
            if not isinstance(parts, Sequence):
                raise TypeError("message content must be a string or sequence")
            for part in parts:
                _append_message_part(values, types, images, part)
        for call in message.get("tool_calls", ()) or ():
            if not isinstance(call, Mapping):
                raise TypeError("tool_calls must contain mappings")
            _append_tool_call(values, types, call)
        if message.get("tool_result") is not None:
            values.append(DEFAULT_SPECIAL_TOKENS["<|tool_result|>"])
            types.append(int(AtomType.LANGUAGE_TOKEN))
            _append_json(values, types, message["tool_result"])
        values.append(DEFAULT_SPECIAL_TOKENS["<|eom|>"])
        types.append(int(AtomType.LANGUAGE_TOKEN))
    values.append(DEFAULT_SPECIAL_TOKENS["<eos>"])
    types.append(int(AtomType.LANGUAGE_TOKEN))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    producer_info = dict(producer or {})
    producer_info.setdefault("protocol", "messages-v1")
    with UnifiedDatasetWriter(
        output,
        tokenizer={
            "name": "robofm-byte-v1",
            "byte_token_base": BYTE_TOKEN_BASE,
            "vocab_size": TOKENIZER_VOCAB_SIZE,
        },
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        producer=producer_info,
        max_atoms_per_shard=max_atoms_per_shard,
    ) as writer:
        writer.append_record(
            np.asarray(values, dtype=np.uint64),
            np.asarray(types, dtype=np.uint8),
            images=images,
        )
    return output


def write_unified_record(
    output_path: str | Path,
    fields: Mapping[str, Any],
    *,
    producer: Mapping[str, Any] | None = None,
    max_atoms_per_shard: int = 10_000_000,
) -> Path:
    """Write a trajectory as standard tool messages in a V1 dataset."""
    producer_info = dict(producer or {})
    producer_info.setdefault("protocol", "trajectory-messages-v1")
    return write_unified_messages(
        output_path,
        list(trajectory_messages(fields)),
        producer=producer_info,
        max_atoms_per_shard=max_atoms_per_shard,
    )


__all__ = [
    "BYTE_TOKEN_BASE",
    "DEFAULT_SPECIAL_TOKENS",
    "TOKENIZER_VOCAB_SIZE",
    "trajectory_messages",
    "write_unified_messages",
    "write_unified_record",
]
