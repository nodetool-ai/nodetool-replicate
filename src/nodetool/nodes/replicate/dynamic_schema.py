from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from pydantic import Field

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    ImageRef,
    Provider,
    VideoRef,
    asset_types,
)
from nodetool.workflows.base_node import BaseNode, ApiKeyMissingError
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.prediction import Prediction, PredictionResult

CACHE_DIR = Path(
    os.getenv(
        "NODETOOL_REPLICATE_SCHEMA_CACHE",
        os.path.join(Path.home(), ".cache", "nodetool", "replicate_schema"),
    )
)

REPLICATE_API_BASE = "https://api.replicate.com/v1"

REPLICATE_STATUS_MAP = {
    "starting": "starting",
    "succeeded": "completed",
    "processing": "running",
    "failed": "failed",
    "canceled": "canceled",
}


@dataclass(frozen=True)
class ReplicateSchemaBundle:
    model_id: str
    version_id: str
    openapi: dict[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    model_description: str


class DynamicReplicate(BaseNode):
    """
    Dynamic Replicate node for running any model on replicate.com.
    replicate, schema, dynamic, inference, runtime, model, api
    Use cases:
    - Run any Replicate model without adding new Python nodes
    - Prototype workflows with new models by pasting a model identifier
    - Build flexible pipelines that depend on runtime model selection
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True
    _dynamic_input_types: dict[str, TypeMetadata] = {}

    model_info: str = Field(
        default="",
        description=(
            "Paste a Replicate model identifier (e.g. runwayml/gen-4.5) "
            "or URL (e.g. https://replicate.com/runwayml/gen-4.5)"
        ),
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._dynamic_input_types = {}
        self._prime_schema_outputs()

    @classmethod
    def get_node_type(cls) -> str:
        return "replicate.DynamicReplicate"

    @classmethod
    def get_namespace(cls) -> str:
        return "replicate"

    @classmethod
    def get_basic_fields(cls):
        return ["model_info"]

    @classmethod
    def _cache_dir(cls) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR

    def _prime_schema_outputs(self) -> None:
        owner, name = _normalize_model_info(self.model_info)
        if not owner or not name:
            return
        cache_key = _cache_key(owner, name)
        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            bundle = _parse_cached_schema(cached, owner, name)
            self._set_dynamic_outputs(bundle)
            self._set_dynamic_properties(bundle)

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        bundle = await self._load_schema_bundle(context)
        self._set_dynamic_outputs(bundle)
        self._set_dynamic_properties(bundle)

        input_values = dict(self.dynamic_properties)
        arguments = await _build_arguments(bundle.input_schema, input_values, context)

        result = await self._run_prediction(context, bundle, arguments)

        outputs = _map_output_values(bundle.output_schema, result)
        if not outputs:
            return {"output": result}
        return outputs

    def find_property(self, name: str):
        from nodetool.workflows.property import Property

        if name in self._dynamic_input_types:
            return Property(name=name, type=self._dynamic_input_types[name])
        return super().find_property(name)

    def find_output_instance(self, name: str):
        slot = super().find_output_instance(name)
        if slot is not None:
            return slot
        if self._supports_dynamic_outputs:
            return _make_output_slot(name, TypeMetadata(type="any"))
        return None

    async def _load_schema_bundle(
        self, context: ProcessingContext
    ) -> ReplicateSchemaBundle:
        owner, name = _normalize_model_info(self.model_info)
        if not owner or not name:
            raise ValueError(
                "model_info must be a Replicate model identifier "
                "(e.g. runwayml/gen-4.5) or URL"
            )

        cache_key = _cache_key(owner, name)
        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            return _parse_cached_schema(cached, owner, name)

        token = await context.get_secret("REPLICATE_API_TOKEN")
        if not token:
            raise ApiKeyMissingError("REPLICATE_API_TOKEN is not configured")

        model_data = await _fetch_model_info(owner, name, token)
        _save_cached_schema(self._cache_dir(), cache_key, model_data)
        return _parse_model_data(model_data, owner, name)

    async def _run_prediction(
        self,
        context: ProcessingContext,
        bundle: ReplicateSchemaBundle,
        arguments: dict[str, Any],
    ) -> Any:
        model_with_version = f"{bundle.model_id}:{bundle.version_id}"

        token = await context.get_secret("REPLICATE_API_TOKEN")
        if token and "REPLICATE_API_TOKEN" not in context.environment:
            context.environment["REPLICATE_API_TOKEN"] = token

        return await context.run_prediction(
            provider=Provider.Replicate,
            node_id=self.id,
            model=model_with_version,
            params=arguments,
            run_prediction_function=_run_replicate_dynamic,
        )

    def _set_dynamic_outputs(self, bundle: ReplicateSchemaBundle) -> None:
        outputs = _build_output_types(bundle.output_schema, bundle.model_description)
        if not outputs:
            outputs = {"output": TypeMetadata(type="any")}
        self._dynamic_outputs = outputs

    def _set_dynamic_properties(self, bundle: ReplicateSchemaBundle) -> None:
        schema_props = bundle.input_schema.get("properties", {})
        required = set(bundle.input_schema.get("required", []))

        self._dynamic_input_types = {}

        for prop_name, prop_schema in schema_props.items():
            input_type = _infer_input_type(prop_schema)
            self._dynamic_input_types[prop_name] = input_type

            if prop_name not in self._dynamic_properties:
                self._dynamic_properties[prop_name] = _default_value(
                    prop_schema, required=(prop_name in required), prop_name=prop_name
                )


# ---------------------------------------------------------------------------
# Prediction runner for dynamic nodes
# ---------------------------------------------------------------------------


async def _run_replicate_dynamic(
    prediction: Prediction, env: dict[str, str]
) -> AsyncGenerator[Prediction | PredictionResult, None]:
    import replicate as replicate_sdk

    model_id = prediction.model
    params = prediction.params
    assert model_id, "Model not found"

    token = env.get("REPLICATE_API_TOKEN")
    if not token:
        raise ApiKeyMissingError("REPLICATE_API_TOKEN is not configured")

    client = replicate_sdk.Client(token)
    started_at = datetime.now()

    match = re.match(r"^(?P<owner>[^/]+)/(?P<name>[^:]+):(?P<version>.+)$", model_id)
    if match:
        replicate_pred = client.predictions.create(
            version=match.group("version"),
            input=params,
        )
    else:
        replicate_pred = client.predictions.create(
            model=model_id,
            input=params,
        )

    current_status = "starting"

    for _ in range(1800):
        replicate_pred = client.predictions.get(replicate_pred.id)

        if replicate_pred.status != current_status:
            current_status = replicate_pred.status
            prediction.status = REPLICATE_STATUS_MAP.get(
                replicate_pred.status, replicate_pred.status
            )

        prediction.logs = replicate_pred.logs
        prediction.error = replicate_pred.error
        prediction.duration = (datetime.now() - started_at).total_seconds()
        yield prediction

        if replicate_pred.status in ("failed", "canceled"):
            raise ValueError(replicate_pred.error or "Prediction failed")

        if replicate_pred.status == "succeeded":
            break

        await asyncio.sleep(1)

    yield PredictionResult(
        prediction=prediction,
        encoding="json",
        content=replicate_pred.output,
    )


# ---------------------------------------------------------------------------
# Schema fetching & caching
# ---------------------------------------------------------------------------


async def _fetch_model_info(owner: str, name: str, token: str) -> dict[str, Any]:
    url = f"{REPLICATE_API_BASE}/models/{owner}/{name}"
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        return resp.json()


def _cache_key(owner: str, name: str) -> str:
    return hashlib.sha256(f"{owner}/{name}".encode("utf-8")).hexdigest()


def _load_cached_schema(cache_dir: Path, cache_key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cached_schema(cache_dir: Path, cache_key: str, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


def _parse_model_data(
    model_data: dict[str, Any], owner: str, name: str
) -> ReplicateSchemaBundle:
    latest = model_data.get("latest_version") or {}
    openapi = latest.get("openapi_schema") or {}
    version_id = latest.get("id", "")
    description = model_data.get("description", "")

    schemas = openapi.get("components", {}).get("schemas", {})
    input_schema = schemas.get("Input", {})
    output_schema = schemas.get("Output", {})

    if input_schema.get("properties"):
        resolved_props = {}
        for prop_name, prop_schema in input_schema["properties"].items():
            resolved_props[prop_name] = _resolve_ref(openapi, prop_schema)
        input_schema = {**input_schema, "properties": resolved_props}

    return ReplicateSchemaBundle(
        model_id=f"{owner}/{name}",
        version_id=version_id,
        openapi=openapi,
        input_schema=input_schema,
        output_schema=_resolve_ref(openapi, output_schema),
        model_description=description,
    )


def _parse_cached_schema(
    cached: dict[str, Any], owner: str, name: str
) -> ReplicateSchemaBundle:
    return _parse_model_data(cached, owner, name)


def _resolve_ref(openapi: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path.lstrip("#/").split("/")
            current: Any = openapi
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part, {})
                else:
                    return schema
            if isinstance(current, dict):
                return _resolve_ref(openapi, current)
        return schema
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for entry in schema["allOf"]:
            resolved = _resolve_ref(openapi, entry)
            if "properties" in resolved:
                merged.setdefault("properties", {}).update(resolved["properties"])
            if "required" in resolved:
                existing = merged.get("required", [])
                merged["required"] = list(set(existing + resolved["required"]))
            for k, v in resolved.items():
                if k not in ("properties", "required") and k not in merged:
                    merged[k] = v
        return merged
    if "anyOf" in schema:
        options = schema.get("anyOf", [])
        if options:
            return _resolve_ref(openapi, options[0])
    if "oneOf" in schema:
        options = schema.get("oneOf", [])
        if options:
            return _resolve_ref(openapi, options[0])
    return schema


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _normalize_model_info(model_info: str) -> tuple[str | None, str | None]:
    raw = model_info.strip()
    if not raw:
        return None, None

    url_match = re.match(r"https?://(?:www\.)?replicate\.com/([^/]+)/([^/?#]+)", raw)
    if url_match:
        return url_match.group(1), url_match.group(2)

    parts = raw.split(":")
    id_part = parts[0]
    segments = id_part.split("/")
    if len(segments) == 2:
        owner = segments[0].strip()
        name = segments[1].strip()
        if owner and name and " " not in owner and " " not in name:
            return owner, name

    return None, None


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def _infer_input_type(prop_schema: dict[str, Any]) -> TypeMetadata:
    if "enum" in prop_schema:
        return TypeMetadata(type="enum", values=prop_schema["enum"])

    kind = prop_schema.get("type")
    fmt = prop_schema.get("format", "")

    if kind == "string" and fmt in ("uri", "url"):
        desc = prop_schema.get("description", "").lower()
        title = prop_schema.get("title", "").lower()
        if _text_suggests_video(desc, title):
            return TypeMetadata(type="video")
        if _text_suggests_audio(desc, title):
            return TypeMetadata(type="audio")
        return TypeMetadata(type="image")

    if kind == "array":
        items = prop_schema.get("items", {})
        item_fmt = items.get("format", "")
        if item_fmt in ("uri", "url"):
            return TypeMetadata(type="list", type_args=[TypeMetadata(type="image")])
        return TypeMetadata(type="list", type_args=[TypeMetadata(type="any")])

    if kind == "string":
        return TypeMetadata(type="str")
    if kind == "integer":
        return TypeMetadata(type="int")
    if kind == "number":
        return TypeMetadata(type="float")
    if kind == "boolean":
        return TypeMetadata(type="bool")

    return TypeMetadata(type="any")


def _infer_output_type(
    output_schema: dict[str, Any], model_description: str
) -> TypeMetadata:
    kind = output_schema.get("type")
    fmt = output_schema.get("format", "")

    if kind == "string" and fmt in ("uri", "url"):
        desc_lower = model_description.lower()
        if _text_suggests_video(desc_lower):
            return TypeMetadata(type="video")
        if _text_suggests_audio(desc_lower):
            return TypeMetadata(type="audio")
        return TypeMetadata(type="image")

    if kind == "string":
        return TypeMetadata(type="str")

    if kind == "array":
        items = output_schema.get("items", {})
        item_fmt = items.get("format", "")
        item_type = items.get("type", "")
        if item_fmt in ("uri", "url") or item_type == "string":
            desc_lower = model_description.lower()
            if _text_suggests_video(desc_lower):
                return TypeMetadata(type="list", type_args=[TypeMetadata(type="video")])
            if _text_suggests_audio(desc_lower):
                return TypeMetadata(type="list", type_args=[TypeMetadata(type="audio")])
            return TypeMetadata(type="list", type_args=[TypeMetadata(type="image")])
        return TypeMetadata(type="list", type_args=[TypeMetadata(type="any")])

    if kind == "object":
        return TypeMetadata(type="dict")
    if kind == "integer":
        return TypeMetadata(type="int")
    if kind == "number":
        return TypeMetadata(type="float")
    if kind == "boolean":
        return TypeMetadata(type="bool")

    return TypeMetadata(type="any")


def _text_suggests_video(*texts: str) -> bool:
    combined = " ".join(texts).lower()
    return "video" in combined or "mp4" in combined


def _text_suggests_audio(*texts: str) -> bool:
    combined = " ".join(texts).lower()
    return "audio" in combined or "music" in combined or "speech" in combined


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def _default_value(
    prop_schema: dict[str, Any],
    *,
    required: bool = False,
    prop_name: str | None = None,
) -> Any:
    if "default" in prop_schema:
        return prop_schema["default"]
    kind = prop_schema.get("type")
    if prop_name == "seed" and kind in ("integer", "number") and not required:
        return -1
    if kind == "string":
        return "" if required else None
    if kind == "integer" or kind == "number":
        return 0 if required else None
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    return None


# ---------------------------------------------------------------------------
# Argument building
# ---------------------------------------------------------------------------


async def _build_arguments(
    input_schema: dict[str, Any],
    input_values: dict[str, Any],
    context: ProcessingContext,
) -> dict[str, Any]:
    schema_props = input_schema.get("properties", {})
    required_props = set(input_schema.get("required", []))
    arguments: dict[str, Any] = {}

    for prop_name, prop_schema in schema_props.items():
        value = input_values.get(prop_name)
        if value is None:
            if prop_name in required_props:
                raise ValueError(f"Missing required input: {prop_name}")
            continue
        if prop_name == "seed" and value in (0, -1) and prop_name not in required_props:
            continue
        arguments[prop_name] = await _coerce_value(prop_schema, value, context)

    return arguments


async def _coerce_value(
    prop_schema: dict[str, Any], value: Any, context: ProcessingContext
) -> Any:
    asset_ref = _try_as_asset_ref(value)
    if asset_ref is not None:
        return await _serialize_asset_ref(asset_ref, context)

    if isinstance(value, list):
        return [
            await _coerce_value(prop_schema.get("items", {}), v, context)
            for v in value
            if v is not None
        ]

    return value


def _try_as_asset_ref(value: Any) -> AssetRef | None:
    if isinstance(value, AssetRef):
        return value
    if isinstance(value, dict) and value.get("type") in asset_types:
        try:
            return AssetRef.from_dict(value)
        except Exception:
            return None
    return None


async def _serialize_asset_ref(asset_ref: AssetRef, context: ProcessingContext) -> str:
    if asset_ref.uri and asset_ref.uri.startswith(("http://", "https://", "data:")):
        return asset_ref.uri
    if isinstance(asset_ref, ImageRef):
        image_base64 = await context.image_to_base64(asset_ref)
        return f"data:image/png;base64,{image_base64}"
    asset_bytes = await context.asset_to_bytes(asset_ref)
    if isinstance(asset_ref, AudioRef):
        import base64

        return f"data:audio/wav;base64,{base64.b64encode(asset_bytes).decode()}"
    if isinstance(asset_ref, VideoRef):
        import base64

        return f"data:video/mp4;base64,{base64.b64encode(asset_bytes).decode()}"
    import base64

    return (
        f"data:application/octet-stream;base64,{base64.b64encode(asset_bytes).decode()}"
    )


# ---------------------------------------------------------------------------
# Output mapping
# ---------------------------------------------------------------------------


def _build_output_types(
    output_schema: dict[str, Any], model_description: str
) -> dict[str, TypeMetadata]:
    kind = output_schema.get("type")
    fmt = output_schema.get("format", "")

    if kind == "string" and fmt in ("uri", "url"):
        out_type = _infer_output_type(output_schema, model_description)
        return {"output": out_type}

    if kind == "string":
        return {"output": TypeMetadata(type="str")}

    if kind == "array":
        out_type = _infer_output_type(output_schema, model_description)
        return {"output": out_type}

    if kind == "object":
        properties = output_schema.get("properties", {})
        if properties:
            result: dict[str, TypeMetadata] = {}
            for name, schema in properties.items():
                result[name] = _infer_output_type(schema, model_description)
            return result
        return {"output": TypeMetadata(type="dict")}

    return {"output": _infer_output_type(output_schema, model_description)}


def _map_output_values(output_schema: dict[str, Any], result: Any) -> dict[str, Any]:
    kind = output_schema.get("type")
    fmt = output_schema.get("format", "")

    if kind == "string" and fmt in ("uri", "url"):
        if isinstance(result, str):
            ref = _uri_to_asset_ref(result)
            return {"output": ref}

    if kind == "array" and isinstance(result, list):
        mapped = [
            _uri_to_asset_ref(item) if isinstance(item, str) else item
            for item in result
        ]
        return {"output": mapped}

    if isinstance(result, str):
        if result.startswith("http") and _looks_like_file_url(result):
            return {"output": _uri_to_asset_ref(result)}
        return {"output": result}

    if isinstance(result, list):
        mapped = [
            (
                _uri_to_asset_ref(item)
                if isinstance(item, str) and _looks_like_file_url(item)
                else item
            )
            for item in result
        ]
        return {"output": mapped}

    return {"output": result}


def _uri_to_asset_ref(uri: str) -> AssetRef:
    lower = uri.lower()
    if any(ext in lower for ext in (".mp4", ".mov", ".webm", ".avi")):
        return VideoRef(uri=uri)
    if any(ext in lower for ext in (".mp3", ".wav", ".flac", ".ogg", ".aac")):
        return AudioRef(uri=uri)
    if any(ext in lower for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        return ImageRef(uri=uri)
    return ImageRef(uri=uri)


def _looks_like_file_url(url: str) -> bool:
    return url.startswith("http") and (
        "replicate.delivery" in url
        or "replicate.com" in url
        or any(
            ext in url.lower()
            for ext in (
                ".mp4",
                ".mov",
                ".webm",
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".mp3",
                ".wav",
                ".flac",
            )
        )
    )


def _make_output_slot(name: str, type_metadata: TypeMetadata):
    from nodetool.metadata.types import OutputSlot

    return OutputSlot(type=type_metadata, name=name)


# ---------------------------------------------------------------------------
# Public API for schema resolution (called by backend endpoint)
# ---------------------------------------------------------------------------


async def resolve_dynamic_schema(
    model_info: str, api_token: str = ""
) -> dict[str, Any]:
    """
    Resolve the Replicate model schema from a model identifier.
    Returns a dict the UI can use to populate the DynamicReplicate node.
    """
    owner, name = _normalize_model_info(model_info)
    if not owner or not name:
        raise ValueError(
            "model_info must be a Replicate model identifier "
            "(e.g. runwayml/gen-4.5) or URL"
        )

    cache_dir = DynamicReplicate._cache_dir()
    ck = _cache_key(owner, name)
    cached = _load_cached_schema(cache_dir, ck)

    if cached is not None:
        bundle = _parse_cached_schema(cached, owner, name)
    else:
        if not api_token:
            api_token = os.environ.get("REPLICATE_API_TOKEN", "")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required to fetch model schemas")

        model_data = await _fetch_model_info(owner, name, api_token)
        _save_cached_schema(cache_dir, ck, model_data)
        bundle = _parse_model_data(model_data, owner, name)

    return _bundle_to_resolve_result(bundle)


def _bundle_to_resolve_result(bundle: ReplicateSchemaBundle) -> dict[str, Any]:
    schema_props = bundle.input_schema.get("properties", {})
    required = set(bundle.input_schema.get("required", []))

    dynamic_properties = {
        prop_name: _default_value(
            prop_schema, required=(prop_name in required), prop_name=prop_name
        )
        for prop_name, prop_schema in schema_props.items()
    }

    dynamic_inputs: dict[str, Any] = {}
    for prop_name, prop_schema in schema_props.items():
        meta = _infer_input_type(prop_schema)
        entry = _type_metadata_to_dict(meta)
        desc = prop_schema.get("description")
        if desc is not None:
            entry["description"] = desc
        if prop_name in dynamic_properties:
            entry["default"] = dynamic_properties[prop_name]
        if "enum" in prop_schema:
            entry["values"] = prop_schema["enum"]
        if meta.type in ("int", "float"):
            if "minimum" in prop_schema:
                entry["min"] = prop_schema["minimum"]
            if "maximum" in prop_schema:
                entry["max"] = prop_schema["maximum"]
        dynamic_inputs[prop_name] = entry

    output_types = _build_output_types(bundle.output_schema, bundle.model_description)
    dynamic_outputs = {
        out_name: _type_metadata_to_dict(meta)
        for out_name, meta in output_types.items()
    }
    if not dynamic_outputs:
        dynamic_outputs = {"output": _type_metadata_to_dict(TypeMetadata(type="any"))}

    return {
        "model_id": bundle.model_id,
        "dynamic_properties": dynamic_properties,
        "dynamic_inputs": dynamic_inputs,
        "dynamic_outputs": dynamic_outputs,
    }


def _type_metadata_to_dict(meta: TypeMetadata) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": meta.type,
        "type_args": [],
        "optional": getattr(meta, "optional", False),
    }
    if getattr(meta, "values", None):
        out["values"] = meta.values
    if getattr(meta, "type_name", None):
        out["type_name"] = meta.type_name
    if getattr(meta, "type_args", None):
        out["type_args"] = [
            _type_metadata_to_dict(a) if isinstance(a, TypeMetadata) else a
            for a in meta.type_args
        ]
    return out
