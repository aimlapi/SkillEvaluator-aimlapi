# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AI/ML API provider: fixed endpoint, own credential, and no credential leaks."""

from __future__ import annotations

import json

import httpx
import pytest

from skillevaluator import model_catalog
from skillevaluator.inference.client import LLMClient
from skillevaluator.inference.diagnostics import safe_llm_labels
from skillevaluator.model_catalog import fetch_model_records
from skillevaluator.provider_config import (
    AIMLAPI_BASE_URL,
    ProviderConfigurationError,
    resolve_embedding_provider,
    resolve_llm_provider,
)
from skillevaluator.validators.security import _skillspector_provider_env

_KEY = "aimlapi-test-key"


def test_aimlapi_resolves_fixed_endpoint_and_own_credential() -> None:
    config = resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "aimlapi", "AIMLAPI_API_KEY": _KEY})

    assert config.provider == "aimlapi"
    assert config.base_url == "https://api.aimlapi.com/v1" == AIMLAPI_BASE_URL
    assert config.api_key == _KEY
    assert config.model == "openai/gpt-6.1-sol"
    assert config.litellm_model == "openai/openai/gpt-6.1-sol"
    assert config.child_environment() == {"AIMLAPI_API_KEY": _KEY}


def test_aimlapi_ignores_endpoint_overrides_so_the_key_stays_on_its_host() -> None:
    env = {
        "SKILL_EVAL_LLM_PROVIDER": "aimlapi",
        "AIMLAPI_API_KEY": _KEY,
        "SKILL_EVAL_LLM_BASE_URL": "https://evil.example/v1",
        "OPENAI_BASE_URL": "https://api.aimlapi.com.evil.example/v1",
        "SKILL_EVAL_EMBEDDING_BASE_URL": "https://evil.example/v1",
    }

    assert resolve_llm_provider(env).base_url == AIMLAPI_BASE_URL
    assert resolve_embedding_provider(env).base_url == AIMLAPI_BASE_URL


def test_aimlapi_requires_its_own_key_and_never_borrows_another() -> None:
    with pytest.raises(ProviderConfigurationError, match="AIMLAPI_API_KEY is required"):
        resolve_llm_provider(
            {
                "SKILL_EVAL_LLM_PROVIDER": "aimlapi",
                "OPENAI_API_KEY": "openai-key",
                "SKILL_EVAL_LLM_API_KEY": "gateway-key",
            }
        )


def test_aimlapi_embedding_defaults_and_model_override() -> None:
    env = {"SKILL_EVAL_LLM_PROVIDER": "aimlapi", "AIMLAPI_API_KEY": _KEY}
    assert resolve_embedding_provider(env).model == "openai/text-embedding-3-small"

    env["SKILL_EVAL_EMBEDDING_MODEL"] = "openai/text-embedding-3-large"
    config = resolve_embedding_provider(env)
    assert (config.model, config.base_url, config.api_key) == (
        "openai/text-embedding-3-large",
        AIMLAPI_BASE_URL,
        _KEY,
    )


def test_aimlapi_model_override_is_kept() -> None:
    config = resolve_llm_provider(
        {"SKILL_EVAL_LLM_PROVIDER": "aimlapi", "AIMLAPI_API_KEY": _KEY, "SKILL_EVAL_LLM_MODEL": "openai/gpt-5.4-mini"}
    )
    assert config.model == "openai/gpt-5.4-mini"


def test_aimlapi_label_is_reportable() -> None:
    assert safe_llm_labels("aimlapi", "openai/gpt-6.1-sol") == ("aimlapi", "openai/gpt-6.1-sol")


def test_aimlapi_catalog_request_goes_only_to_aimlapi_host(monkeypatch) -> None:
    seen: list[tuple[str, str | None]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            raw = json.dumps({"data": [{"id": "openai/gpt-6.1-sol"}]}).encode()
            return raw if size < 0 else raw[:size]

    def fake_urlopen(request, *, timeout):
        seen.append((request.full_url, request.get_header("Authorization")))
        return _Response()

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)
    config = resolve_llm_provider({"SKILL_EVAL_LLM_PROVIDER": "aimlapi", "AIMLAPI_API_KEY": _KEY})

    records = fetch_model_records(config)

    assert [record.id for record in records] == ["openai/gpt-6.1-sol"]
    assert seen == [("https://api.aimlapi.com/v1/models", f"Bearer {_KEY}")]


def test_aimlapi_chat_key_is_sent_to_api_host_and_dropped_on_cross_host_redirect(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.aimlapi.com":
            return httpx.Response(307, headers={"Location": "https://evil.example/v1/chat/completions"})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-6.1-sol",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    for name in ("SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL", "SKILL_EVAL_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "aimlapi")
    monkeypatch.setenv("AIMLAPI_API_KEY", _KEY)
    http_client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    client = LLMClient(http_client=http_client, max_retries=0)

    client.chat([{"role": "user", "content": "hi"}])

    assert client.provider == "aimlapi"
    first, redirected = requests
    assert (first.url.scheme, first.url.host, first.url.path) == ("https", "api.aimlapi.com", "/v1/chat/completions")
    assert first.headers["Authorization"] == f"Bearer {_KEY}"
    assert redirected.url.host == "evil.example"
    assert "authorization" not in {name.lower() for name in redirected.headers}
    assert _KEY not in str(redirected.headers)
    assert not any(name.lower().startswith("x-aimlapi") for request in requests for name in request.headers)


def test_skillspector_bridge_routes_aimlapi_through_openai_adapter(monkeypatch) -> None:
    for name in ("SKILLSPECTOR_PROVIDER", "SKILLSPECTOR_MODEL", "SKILL_EVAL_LLM_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "aimlapi")
    monkeypatch.setenv("AIMLAPI_API_KEY", _KEY)

    child_env, provider = _skillspector_provider_env()

    assert provider == "aimlapi"
    assert child_env["SKILLSPECTOR_PROVIDER"] == "openai"
    assert child_env["SKILLSPECTOR_MODEL"] == "openai/gpt-6.1-sol"
    assert child_env["OPENAI_API_KEY"] == _KEY
    assert child_env["OPENAI_BASE_URL"] == AIMLAPI_BASE_URL
