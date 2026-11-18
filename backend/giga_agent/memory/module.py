from __future__ import annotations

import re
from html import escape
from pathlib import PurePosixPath
from typing import Any, List, Optional

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from giga_agent.conf import get_settings
from giga_agent.core.agent.base import BaseAgent
from giga_agent.core.agent.middleware import AgentMiddleware
from giga_agent.core.agent.runtime_resolver import RuntimeResolver
from giga_agent.core.agent.types import AgentState
from giga_agent.core.logging import get_logger
from giga_agent.core.module import BaseModule
from giga_agent.memory.backends.base import MemoryFileDTO
from giga_agent.memory.events import ensure_subscribed
from giga_agent.memory.middleware import MemoryMiddleware
from giga_agent.memory.paths import (
    global_about_path,
    is_about_file,
    parse_memory_path,
    tagged_about_path,
)
from giga_agent.memory.runtime import (
    build_memory_service,
    get_memory_show_global,
    get_memory_tags,
    is_memory_disabled,
)
from giga_agent.memory.service import MemoryService
from giga_agent.memory.tool import search_memories
from giga_agent.models.users import UserShort
from giga_agent.modules.io.memory_bridge import (
    ABOUT_GLOBAL_TEMPLATE,
    ABOUT_TAGGED_TEMPLATE,
)


logger = get_logger(__name__)


SENDER_FILE_LIMIT = 2
FILE_CHAR_LIMIT = 2000
ABOUT_CHAR_LIMIT = 8000


class MemoryModule(BaseModule):
    id: str = "memory"

    async def _has_embeddings(self, user: UserShort | None, *, config) -> bool:
        if config is None:
            return False
        try:
            resolver = RuntimeResolver.from_config(config)
        except ValueError:
            try:
                resolver = await RuntimeResolver.create(config)
                resolver.inject(config)
            except Exception:
                return False
        return resolver.has_embedding

    async def get_tools(
        self,
        user: UserShort | None,
        agent: BaseAgent,
        *,
        config=None,
        **kwargs: Any,
    ) -> List[BaseTool]:
        _ = user, agent, kwargs
        if config is None:
            return []
        if is_memory_disabled(config):
            return []
        if not await self._has_embeddings(user, config=config):
            return []
        return [search_memories]

    async def get_instructions(
        self,
        user: UserShort | None,
        agent: BaseAgent,
        state: Optional["AgentState"] = None,
        config=None,
        **kwargs: Any,
    ) -> str | None:
        _ = user, agent, kwargs
        if config is None:
            return None
        if is_memory_disabled(config):
            return None

        tags = get_memory_tags(config)
        show_global = get_memory_show_global(config, tags=tags)

        try:
            service = await build_memory_service(config, include_fast_llm=False)
        except Exception:
            logger.exception("MemoryModule.get_instructions: build failed")
            return None

        try:
            files = await service.list_visible(
                visible_tags=tags or None,
                include_global=show_global,
            )
        except Exception:
            logger.exception("MemoryModule.get_instructions: list failed")
            return None

        configurable = config.get("configurable") or {}
        query = configurable.get("memory_query")
        if not isinstance(query, str):
            query = next(
                (
                    str(m.text)
                    for m in reversed((state or {}).get("messages", []))
                    if isinstance(m, HumanMessage)
                ),
                "",
            )
        return await _render_memory_block(
            service=service,
            files=files,
            query=query.strip()[:4000],
            username=configurable.get("memory_username"),
            tags=tags,
            show_global=show_global,
        )

    def get_middleware(self, **kwargs: Any) -> AgentMiddleware:
        _ = kwargs
        return MemoryMiddleware()

    def get_api_router(self, **kwargs: Any) -> APIRouter:
        _ = kwargs
        from giga_agent.memory.api import router

        return router

    async def on_startup(self, session, **kwargs: Any) -> None:
        _ = session, kwargs
        await ensure_subscribed()


def _sender_files(
    files: list[MemoryFileDTO], username: str | None
) -> list[tuple[MemoryFileDTO, str]]:
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_]+", username):
        return []
    nick = username.lower()
    filename_match = re.compile(rf"(?:^|[-_.@]){re.escape(nick)}(?:$|[-_.])", re.I)
    mention = re.compile(rf"(?<![a-z0-9_])@{re.escape(nick)}(?![a-z0-9_])", re.I)
    ranked = []
    for file in files:
        name = PurePosixPath(file.path).stem.lower()
        anchor = ""
        if name in (nick, f"profile_{nick}", f"@{nick}"):
            rank = 0
        elif not name.startswith("profile_") and filename_match.search(name):
            rank = 1
        elif mention.search(file.description or ""):
            rank = 2
        elif match := mention.search(file.content):
            rank = 3
            anchor = match.group()
        else:
            continue
        ranked.append((rank, file.path, file, anchor))
    return [(file, anchor) for _, _, file, anchor in sorted(ranked)[:SENDER_FILE_LIMIT]]


def _excerpt(content: str, limit: int, anchor: str = "") -> str:
    if len(content) <= limit:
        return content.strip()
    pos = content.find(anchor) if anchor else 0
    start = min(max(0, pos - limit // 4), len(content) - limit)
    return (
        ("… " if start else "")
        + content[start : start + limit].strip()
        + (" …" if start + limit < len(content) else "")
    )


async def _render_memory_block(
    *,
    service: MemoryService,
    files: list[MemoryFileDTO],
    query: str,
    username: str | None,
    tags: list[str],
    show_global: bool,
) -> str:
    parts = [
        "<memory>",
        "ABOUT.md — главный файл памяти. Сверяйся с ним в первую очередь. "
        "Храни в нём устойчивые сведения и ссылки на тематические файлы. "
        "Дальше приведена подборка по текущему запросу и собеседнику. "
        "Если для ответа нужны подробности, загрузи в контекст конкретный файл "
        "через read_file по пути из ABOUT.md или подборки.",
    ]
    if service.has_embeddings:
        parts.append(
            "Если путь неизвестен, найди его через search_memories и прочитай нужный файл."
        )
    parts.append(
        "Обновляй память через write_file/edit_file. Каждый файл кроме ABOUT.md "
        "должен иметь YAML-frontmatter с полем description (одна строка, до 200 символов). "
        "Лимит файла — 200 строк."
    )
    abouts = {}
    candidates = {}
    for file in files:
        if is_about_file(parse_memory_path(file.path)):
            abouts[file.tag] = file
        else:
            candidates[file.path] = file

    for tag in ([None] if show_global else []) + tags:
        file = abouts.get(tag)
        if file:
            path, content = file.path, file.content
        elif tag is None:
            path, content = global_about_path(), ABOUT_GLOBAL_TEMPLATE
        else:
            path, content = (
                tagged_about_path(tag),
                ABOUT_TAGGED_TEMPLATE.format(tag=tag),
            )
        parts.append(
            f'<about path="{escape(path)}">\n{_excerpt(content, ABOUT_CHAR_LIMIT)}\n</about>'
        )

    selected = _sender_files(list(candidates.values()), username)
    for file, _ in selected:
        del candidates[file.path]

    top_n = get_settings().giga_agent_memory_prompt_top_n
    semantic_hits = []
    if top_n and candidates and query and service.has_embeddings:
        try:
            semantic_hits = await service.search(
                query,
                n=top_n,
                visible_tags=tags or None,
                include_global=show_global,
                distinct_files=True,
                exclude_paths=tuple(f.path for f in abouts.values())
                + tuple(f.path for f, _ in selected),
            )
        except Exception:
            logger.exception("Memory prompt semantic search failed")

    if semantic_hits:
        for hit in semantic_hits:
            if file := candidates.pop(hit.path, None):
                selected.append((file, hit.snippet))
    elif top_n and query and candidates:
        # Existing installations can have files but no embedding runtime/index.
        terms = set(re.findall(r"[^\W_]{3,}", query.casefold()))
        matches = []
        for file in candidates.values():
            label = (
                f"{PurePosixPath(file.path).stem} {file.description or ''}".casefold()
            )
            score = len(terms.intersection(re.findall(r"[^\W_]+", label)))
            if score:
                matches.append((-score, file.path))
        selected.extend((candidates[path], "") for _, path in sorted(matches)[:top_n])

    for file, anchor in selected:
        parts.append(
            f'<file path="{escape(file.path)}">\n{_excerpt(file.content, FILE_CHAR_LIMIT, anchor)}\n</file>'
        )
    parts.append("</memory>")
    return "\n".join(parts)


__all__ = ["MemoryModule"]
