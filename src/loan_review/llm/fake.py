"""Provide a deterministic, offline LLM stub with strict Pydantic I/O."""

from pydantic import BaseModel, ConfigDict


class FakeLLMRequest(BaseModel):
    """Validate a test prompt without coercion or unexpected fields."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    prompt: str


class FakeLLMResponse(BaseModel):
    """Validate the fixed text returned by the offline stub."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str


class FakeLLM:
    """Return a fixed response without interpreting prompts or making decisions."""

    def invoke(self, request: FakeLLMRequest) -> FakeLLMResponse:
        """Validate the request and return deterministic placeholder text."""
        FakeLLMRequest.model_validate(request)
        return FakeLLMResponse(text="Fake LLM response.")
