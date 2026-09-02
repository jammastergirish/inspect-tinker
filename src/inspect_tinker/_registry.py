"""Registers the ``tinker`` model provider with Inspect AI.

Inspect imports this module via the ``[project.entry-points.inspect_ai]`` entry
point (see pyproject.toml), which runs the ``@modelapi`` decorator below and makes
``get_model("tinker/<checkpoint>")`` resolve to :class:`TinkerAPI`. Importing the
top-level ``inspect_tinker`` package registers it too, as a fallback.
"""

from inspect_ai.model import modelapi


@modelapi(name="tinker")
def tinker():
    from .provider import TinkerAPI

    return TinkerAPI
