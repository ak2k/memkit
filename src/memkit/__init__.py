"""memkit — a tiered markdown memory store's retrieval hook and its checkers.

Three tools over one config file:

- `memory-recall`  the UserPromptSubmit hook (stdlib only, imports under 3.9)
  and its on-demand `--search` mode.
- `memory-integrity`  layout / ledger / frontmatter / link / citation checker
  (3.12+).
- `memory-eval`  snapshot-gated retrieval eval, whose cases are the consumer's
  data and never ship here.

`memkit.memory_prompt_recall` owns the config reader, because it is the module
with the hardest constraints — stdlib only, 3.9-importable, and usable as a
loose file with nothing beside it but its wordlist. The other two import it
rather than growing a second reader to disagree with.
"""

from memkit.memory_prompt_recall import CONFIG_ENV, SCHEMA, ConfigError, load_config

__all__ = ["CONFIG_ENV", "SCHEMA", "ConfigError", "load_config"]
