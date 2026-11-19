"""Shared prompt contract (lillm-summarizer summary-v3) and Granite 4.0 template."""

from __future__ import annotations

MODEL_ID = "ibm-granite/granite-4.0-1b"

# Exact summary-v3 contract from https://github.com/nchapman/lillm-summarizer
PROMPT_VERSION = "summary-v3"
SYSTEM_PROMPT = (
    "Summarize the page below clearly and objectively, capturing its main points "
    "and important supporting details in your own words. Keep simple material "
    "brief, and provide more detail when the source warrants it. State the "
    "substance directly rather than describing the page, attributing opinions "
    "or arguments inline where appropriate. Write in plain prose without markdown, "
    "using multiple paragraphs when helpful, and do not "
    "add unsupported information."
)
USER_PROMPT_TEMPLATE = "{document}"

# Minimal Granite 4.0 chat template matching ibm-granite/granite-4.0-1b's official
# role markers byte-for-byte for system/user/assistant turns without complex Jinja
# methods that break minijinja or prevent prefix/suffix extraction.
GRANITE4_MINIMAL_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif %}"
)

# Eight-question sanity floor gate for verifying converted .litertlm bundles.
FLOOR_GATE_QUESTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("What is the capital of Japan? Answer briefly.", ("tokyo",)),
    ("What is 17 multiplied by 6? Answer with just the number.", ("102",)),
    ("Who wrote the play Romeo and Juliet? Answer briefly.", ("shakespeare",)),
    ("What chemical element has the symbol Au? Answer briefly.", ("gold",)),
    ("How many continents are there on Earth? Answer briefly.", ("7", "seven")),
    ("Which planet is known as the Red Planet? Answer briefly.", ("mars",)),
    ("In what year did humans first land on the Moon? Answer briefly.", ("1969",)),
    ("What is the boiling point of water in degrees Celsius at sea level? Answer briefly.", ("100",)),
]
