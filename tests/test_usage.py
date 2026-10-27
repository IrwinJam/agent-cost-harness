import pytest

from harness.usage import UsageError, from_chat_completions, from_messages, from_responses, normalize


def test_responses_usage():
    u = from_responses(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 800},
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 200},
            "total_tokens": 1300,
        }
    )
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.reasoning_tokens) == (1000, 800, 300, 200)
    assert u.uncached_input_tokens == 200
    assert u.visible_output_tokens == 100
    assert u.cache_hit_rate == pytest.approx(0.8)


def test_responses_missing_details_default_to_zero():
    u = from_responses({"input_tokens": 50, "output_tokens": 5, "input_tokens_details": None})
    assert u.cached_input_tokens == 0 and u.reasoning_tokens == 0


def test_chat_completions_usage():
    u = from_chat_completions(
        {
            "prompt_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 128},
            "completion_tokens": 90,
            "completion_tokens_details": {"reasoning_tokens": 40},
        }
    )
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens, u.reasoning_tokens) == (500, 128, 90, 40)
    assert u.protocol == "chat_completions"


def test_messages_default_assumes_anthropic_convention():
    # Anthropic convention: input_tokens excludes cache reads, so the total adds them back.
    u = from_messages(
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 900,
            "output_tokens": 60,
            "output_tokens_details": {"thinking_tokens": 25},
        }
    )
    assert u.input_tokens == 1000
    assert u.cached_input_tokens == 900
    assert u.reasoning_tokens == 25


def test_messages_inclusive_convention():
    u = from_messages({"input_tokens": 1000, "cache_read_input_tokens": 900, "output_tokens": 60}, input_includes_cached=True)
    assert u.input_tokens == 1000
    assert u.uncached_input_tokens == 100


def test_missing_usage_raises():
    with pytest.raises(UsageError):
        from_responses(None)


def test_inconsistent_usage_raises():
    with pytest.raises(UsageError):
        from_responses({"input_tokens": 10, "input_tokens_details": {"cached_tokens": 11}, "output_tokens": 0})


def test_normalize_dispatch():
    assert normalize("responses", {"input_tokens": 1, "output_tokens": 1}).protocol == "responses"
    with pytest.raises(ValueError):
        normalize("bogus", {})  # type: ignore[arg-type]
