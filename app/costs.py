"""What a model call costs, in US dollars.

OpenAI does not return money. Every response returns token counts (response.usage),
so cost = tokens x price per token. Prices change: check https://openai.com/api/pricing
and update this table when they do, or when you switch models.

Note: OpenAI caches long prompts (1,024+ tokens) and bills the cached part at a lower
price. Our prompts are about 450 tokens, so caching never applies and this is exact.
"""

# USD per 1 million tokens: (input, output). Embedding models produce no output tokens.
PRICES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "text-embedding-3-small": (0.02, 0.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Cost of one call from its token counts."""
    if model not in PRICES_PER_1M_TOKENS:
        raise KeyError(f"No price for model '{model}'. Add it to PRICES_PER_1M_TOKENS in app/costs.py.")
    input_price, output_price = PRICES_PER_1M_TOKENS[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
