"""inspect-tinker — a Tinker-backed model provider for Inspect AI.

Use a Tinker checkpoint as an Inspect model: ``tinker/<tinker-checkpoint-path>``.
Importing this package registers the provider (belt-and-suspenders with the
entry-point registration Inspect discovers automatically).
"""

from ._registry import tinker
from .provider import (
    DEFAULT_BASE_MODEL,
    TinkerAPI,
    message_to_template,
    parse_tool_calls,
    tool_to_schema,
)

__version__ = "0.1.2"
__all__ = [
    "DEFAULT_BASE_MODEL",
    "TinkerAPI",
    "message_to_template",
    "parse_tool_calls",
    "tinker",
    "tool_to_schema",
]
