"""FastAPI service for OCR + LayoutLM analysis."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
import json

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from ..layoutlm.infer import warm_layoutlm_model
from ..layoutlm.token_infer import warm_token_model
from ..io.loaders import load_images_from_bytes
from ..ocr.engine import analyze_pages
from ..pipeline.analyze import analyze_document_pages
from ..regex.classify import infer_present_docs
from ..regex.engine import load_rules, run_rules

LOGGER = logging.getLogger("caesar_ocr.api")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_doc_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket_and_key = uri[5:]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _download_s3_prefix(uri: str, cache_root: Path) -> Path:
    bucket, key_prefix = _parse_s3_uri(uri)
    local_dir = cache_root / bucket / key_prefix
    marker = local_dir / ".complete"
    if marker.exists():
        return local_dir

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    found = False
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found = True
            dest = cache_root / bucket / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
            client.download_file(bucket, key, str(dest))
            downloaded += 1
    if not found:
        raise RuntimeError(f"No objects found under {uri}")
    LOGGER.info("Downloaded %s objects from %s", downloaded, uri)
    local_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")
    return local_dir


def _load_token_model_map() -> Dict[str, str]:
    raw_map = _get_env("CAESAR_S3_TOKEN_MODEL_MAP")
    mapping: Dict[str, str] = {}
    if raw_map:
        try:
            parsed = json.loads(raw_map)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON in CAESAR_S3_TOKEN_MODEL_MAP") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("CAESAR_S3_TOKEN_MODEL_MAP must be a JSON object.")
        mapping = { _normalize_doc_key(k): v for k, v in parsed.items() }
        return mapping

    registry_path = _get_env("CAESAR_TOKEN_MODEL_REGISTRY")
    if registry_path:
        mapping = _load_registry_file(Path(registry_path))
        return mapping

    default_registry = Path(__file__).with_name("token_model_registry.json")
    if default_registry.exists():
        mapping = _load_registry_file(default_registry)
        return mapping

    passport_uri = _get_env("CAESAR_S3_TOKEN_MODEL_PASSPORT")
    if passport_uri:
        mapping["passport"] = passport_uri

    diploma_uri = _get_env("CAESAR_S3_TOKEN_MODEL_DIPLOMA")
    if diploma_uri:
        mapping["degree_certificate"] = diploma_uri

    return mapping


def _load_registry_file(path: Path) -> Dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Registry file not found: {path}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in registry file: {path}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Registry file must be a JSON object: {path}")
    return { _normalize_doc_key(k): v for k, v in parsed.items() }


def _apply_regex_and_hints(ocr_result, *, regex_rules_path: Optional[str], regex_debug: bool) -> None:
    if regex_rules_path:
        rules = load_rules(Path(regex_rules_path))
        regex_fields = run_rules(ocr_result.ocr_text, rules, debug=regex_debug)
        if regex_fields:
            ocr_result.fields.update(regex_fields)
    doc_hints = infer_present_docs(ocr_result.ocr_text)
    if doc_hints:
        ocr_result.fields.setdefault("doc_hints", sorted(doc_hints))
        if ocr_result.doc_type in (None, "", "unknown"):
            ocr_result.doc_type = sorted(doc_hints)[0]


def _warm_models(app: FastAPI) -> None:
    cache_root = Path(_get_env("CAESAR_MODEL_CACHE_DIR") or "/tmp/models")
    token_map = _load_token_model_map()
    if token_map:
        app.state.token_model_uris.update(token_map)
    prefetch_models = _env_bool("CAESAR_PREFETCH_MODELS", default=False)
    eager_warm = _env_bool("CAESAR_WARM_TOKEN_MODELS", default=False)

    if prefetch_models and app.state.token_model_uris:
        for doc_key, uri in app.state.token_model_uris.items():
            local_dir = _download_s3_prefix(uri, cache_root)
            app.state.token_model_paths[doc_key] = str(local_dir)

    token_model_dir = _get_env("CAESAR_LAYOUTLM_TOKEN_MODEL_DIR")
    if token_model_dir:
        try:
            warm_token_model(token_model_dir)
            LOGGER.info("Warm token model loaded from %s", token_model_dir)
        except Exception:
            LOGGER.exception("Failed to warm token model at %s", token_model_dir)
            raise
    elif eager_warm:
        for doc_key, model_dir in app.state.token_model_paths.items():
            try:
                warm_token_model(model_dir)
                LOGGER.info("Warm token model loaded for %s from %s", doc_key, model_dir)
            except Exception:
                LOGGER.exception("Failed to warm token model for %s at %s", doc_key, model_dir)
                raise

    layoutlm_model_dir = _get_env("CAESAR_LAYOUTLM_MODEL_DIR")
    if layoutlm_model_dir:
        try:
            warm_layoutlm_model(layoutlm_model_dir)
            LOGGER.info("Warm LayoutLM model loaded from %s", layoutlm_model_dir)
        except Exception:
            LOGGER.exception("Failed to warm LayoutLM model at %s", layoutlm_model_dir)
            raise


def _resolve_token_model_dir(app: FastAPI, doc_key: str) -> Optional[str]:
    token_model_dir = _get_env("CAESAR_LAYOUTLM_TOKEN_MODEL_DIR")
    if token_model_dir:
        return token_model_dir

    cached = app.state.token_model_paths.get(doc_key)
    if cached:
        return cached

    uri = app.state.token_model_uris.get(doc_key)
    if not uri:
        return None

    cache_root = Path(_get_env("CAESAR_MODEL_CACHE_DIR") or "/tmp/models")
    local_dir = _download_s3_prefix(uri, cache_root)
    app.state.token_model_paths[doc_key] = str(local_dir)
    return str(local_dir)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _warm_models(app)
        yield

    app = FastAPI(title="caesar_ocr API", lifespan=lifespan)
    app.state.token_model_paths = {}
    app.state.token_model_uris = {}

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), doc_hint: str | None = None) -> dict:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file payload.")

        lang = _get_env("CAESAR_OCR_LANG") or "eng+deu"
        pages = load_images_from_bytes(payload, dpi=300)
        ocr_result = analyze_pages(pages, lang=lang)
        _apply_regex_and_hints(
            ocr_result,
            regex_rules_path=_get_env("CAESAR_REGEX_RULES_PATH"),
            regex_debug=_env_bool("CAESAR_REGEX_DEBUG", default=False),
        )

        doc_key = _normalize_doc_key(doc_hint or ocr_result.doc_type or "")
        token_model_dir = _resolve_token_model_dir(app, doc_key)
        layoutlm_model_dir = _get_env("CAESAR_LAYOUTLM_MODEL_DIR")

        result = analyze_document_pages(
            pages,
            ocr_result,
            file_bytes=payload,
            layoutlm_model_dir=layoutlm_model_dir,
            layoutlm_token_model_dir=token_model_dir,
        )
        if result.schema:
            result.schema.ocr.language = lang

        if _env_bool("CAESAR_API_RETURN_SCHEMA", default=False):
            return result.schema.to_dict() if result.schema else result.to_dict(schema=False)

        return {
            "doc_type": result.ocr.doc_type,
            "fields": result.ocr.fields,
            "ocr_text": result.ocr.ocr_text,
        }

    return app


app = create_app()
