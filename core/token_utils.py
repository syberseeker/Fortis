"""
Approximate token counting without an external tokenizer.

Originally this used tiktoken, but tiktoken downloads its BPE encoding file
from openaipublic.blob.core.windows.net on first use -- a hard network
dependency that breaks in exactly the kind of locked-down/offline
environment this tool is meant to run in (and did break, the first time
this repo's test suite was actually run end-to-end).

A ~4-characters-per-token heuristic is standard for English text and close
enough for this tool's purposes: deciding chunk boundaries and deciding
whether an engagement is small enough for a flat analysis pass or needs
map-reduce. Both are approximate budgets, not exact limits, so heuristic
counting is an acceptable trade for removing the network dependency.
"""

_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def tokens_to_chars(n_tokens: int) -> int:
    return n_tokens * _CHARS_PER_TOKEN
