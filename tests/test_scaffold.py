"""Check package installation and the offline stub's public contract."""

from importlib import import_module

import pytest
from pydantic import ValidationError

from loan_review.llm.fake import FakeLLM, FakeLLMRequest, FakeLLMResponse


@pytest.mark.parametrize(
    "subpackage",
    ["data", "model", "policy", "llm", "graph", "audit", "fairness", "evals", "api"],
)
def test_package_imports(subpackage: str) -> None:
    assert import_module(f"loan_review.{subpackage}").__doc__


def test_fake_llm_returns_deterministic_validated_response() -> None:
    llm = FakeLLM()
    request = FakeLLMRequest(prompt="A placeholder test prompt.")

    response = llm.invoke(request)

    assert isinstance(response, FakeLLMResponse)
    assert response.text == "Fake LLM response."
    assert llm.invoke(request) == response


@pytest.mark.parametrize(
    "payload", [{"prompt": 123}, {"prompt": "test", "unexpected": True}]
)
def test_fake_llm_request_rejects_invalid_io(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FakeLLMRequest.model_validate(payload)
