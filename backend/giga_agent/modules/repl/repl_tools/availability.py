"""REPL functions available for the current runtime configuration."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from giga_agent.core.agent.runtime_resolver import RuntimeResolver
from giga_agent.modules.repl.repl_tools.llm import summarize
from giga_agent.modules.repl.repl_tools.sentiment import (
    get_embeddings,
    predict_sentiments,
)


def get_available_repl_tools(
    config: RunnableConfig | None,
) -> list[Callable[..., Awaitable[Any]]]:
    if config is not None:
        try:
            resolver = RuntimeResolver.from_config(config)
        except ValueError:
            pass
        else:
            if resolver.has_embedding:
                return [predict_sentiments, summarize, get_embeddings]
    return [summarize]
