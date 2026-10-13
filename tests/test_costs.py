"""Cost math: tokens x price. No API key needed."""
import pytest

from app.config import settings
from app.costs import PRICES_PER_1M_TOKENS, cost_usd


def test_one_million_input_tokens_costs_the_list_price():
    assert cost_usd("gpt-4o-mini", 1_000_000) == pytest.approx(0.15)


def test_input_and_output_are_priced_separately():
    # 1,000 in x $0.15/M + 100 out x $0.60/M
    assert cost_usd("gpt-4o-mini", 1000, 100) == pytest.approx(0.00021)


def test_unknown_model_fails_loudly_instead_of_costing_zero():
    with pytest.raises(KeyError, match="Add it to PRICES_PER_1M_TOKENS"):
        cost_usd("some-new-model", 100)


def test_every_configured_model_has_a_price():
    # Catches switching a model in .env without updating the price table.
    for model in (settings.embed_model, settings.generation_model, settings.judge_model):
        assert model in PRICES_PER_1M_TOKENS
