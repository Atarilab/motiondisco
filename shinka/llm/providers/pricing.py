# Modified by the MotionDisco authors from ShinkaEvolve (Sakana AI, Apache-2.0); see NOTICE.
"""LLM pricing compatibility API backed by the runtime model catalog."""

from __future__ import annotations

from typing import Optional, Tuple

from shinka.pricing.catalog import ModelPrice, get_catalog


def _entry(model_name: str) -> ModelPrice:
    try:
        return get_catalog().catalog.get_by_name(model_name, kind="llm")
    except KeyError:
        try:
            return get_catalog().catalog.find_by_api_name(model_name, kind="llm")
        except KeyError as exc:
            raise ValueError(f"Model {model_name} not found in pricing data") from exc


def get_model_prices(model_name: str, input_tokens: Optional[int] = None) -> dict:
    input_price, output_price = _entry(model_name).rates(input_tokens)
    if input_price is None or output_price is None:
        raise ValueError(f"Model {model_name} has no pricing data")
    return {"input_price": input_price, "output_price": output_price}


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> Tuple[float, float]:
    prices = get_model_prices(model_name, input_tokens=input_tokens)
    return (
        prices["input_price"] * input_tokens,
        prices["output_price"] * output_tokens,
    )


def model_exists(model_name: str) -> bool:
    try:
        entry = _entry(model_name)
    except ValueError:
        return False
    return entry.input_price is not None and entry.output_price is not None


def is_reasoning_model(model_name: str) -> bool:
    try:
        return _entry(model_name).is_reasoning
    except ValueError:
        return False


def get_provider(model_name: str) -> Optional[str]:
    try:
        return _entry(model_name).provider
    except ValueError:
        return None


def get_all_providers() -> list[str]:
    return get_catalog().catalog.providers(kind="llm")


def get_models_by_provider(provider: str) -> list[str]:
    return get_catalog().catalog.models(provider, kind="llm")


def has_fixed_temperature(model_name: str) -> bool:
    try:
        return _entry(model_name).think_temp_fixed
    except ValueError:
        return False


def requires_reasoning(model_name: str) -> bool:
    try:
        return _entry(model_name).requires_reasoning
    except ValueError:
        return False


# The runtime catalog carries no per-model request-shape flags, so these
# are tracked here. Keyed on API model name.
_TEMPERATURE_UNSUPPORTED_MODELS = frozenset(
    {
        "claude-opus-5",
        "anthropic.claude-opus-5",
        "claude-opus-4-7",
        "anthropic.claude-opus-4-7",
    }
)
_ADAPTIVE_THINKING_MODELS = _TEMPERATURE_UNSUPPORTED_MODELS


def temperature_unsupported(model_name: str) -> bool:
    """Check if a model's API rejects the `temperature` parameter outright.

    Some newer models (e.g. claude-opus-4-7) return a 400 if `temperature`
    is sent at all — for these the field must be omitted from the request.
    """
    return model_name in _TEMPERATURE_UNSUPPORTED_MODELS


def uses_adaptive_thinking(model_name: str) -> bool:
    """Check if an Anthropic model uses adaptive-thinking + output_config.effort.

    Newer Anthropic models (e.g. claude-opus-4-7) reject the legacy
    ``thinking={"type":"enabled","budget_tokens":N}`` shape and require
    ``thinking={"type":"adaptive"}`` plus ``output_config={"effort":...}``.
    """
    return model_name in _ADAPTIVE_THINKING_MODELS
