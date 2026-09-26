"""Small, cached status summaries for active Telegram runs."""

from __future__ import annotations

import json
import time
from typing import Any

from cashews import cache
from langchain_core.messages import HumanMessage, SystemMessage

from giga_agent.channels.telegram.message_tool import TELEGRAM_MESSAGE_TOOL_NAME
from giga_agent.core.agent.runtime_resolver import RuntimeResolver
from giga_agent.core.cache import setup_cache
from giga_agent.core.db import get_session_factory
from giga_agent.core.logging import get_logger
from giga_agent.models.users import UserRepository

logger = get_logger(__name__)

STATUS_TTL_SECONDS = 60
_USER_TEXT_LIMIT = 1200
_STATUS_TEXT_LIMIT = 500
_MAX_TOOLS = 4


def _field(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(_field(part, "text", ""))
            for part in content
            if _field(part, "type") == "text" and _field(part, "text")
        )
    return ""


def _message_tool_call_ids(messages: list[Any]) -> set[str]:
    return {
        str(call_id)
        for message in messages
        if _field(message, "type") == "ai"
        for call in _field(message, "tool_calls", []) or []
        if _field(call, "name") == TELEGRAM_MESSAGE_TOOL_NAME
        if (call_id := _field(call, "id"))
    }


def _user_reply_from_tool(message: Any, message_call_ids: set[str]) -> str | None:
    if _field(message, "type") != "tool":
        return None
    if str(_field(message, "tool_call_id", "")) not in message_call_ids:
        return None
    content = _field(message, "content")
    for part in content if isinstance(content, list) else [content]:
        raw = _field(part, "text") if isinstance(part, dict) else part
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if isinstance(payload, list):
            payload = payload[0] if payload else None
            payload = _field(payload, "text")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    continue
        if not isinstance(payload, dict) or payload.get("channel") != "telegram":
            continue
        if payload.get("kind") != "message_response" or payload.get("auto_response"):
            continue
        answer = payload.get("content")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()[:_USER_TEXT_LIMIT]
    return None


def _user_task(messages: list[Any]) -> tuple[str, int]:
    message_call_ids = _message_tool_call_ids(messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        answer = _user_reply_from_tool(message, message_call_ids)
        if answer:
            return answer, index
        if _field(message, "type") not in ("human", "user"):
            continue
        content = _message_text(_field(message, "content", ""))
        # Telegram wraps the incoming text with sender metadata and instructions.
        if "Текст сообщения:\n" in content:
            content = content.split("Текст сообщения:\n", 1)[1]
            content = content.split("\n\nПрикреплено", 1)[0]
            content = content.split("\n\n[system:", 1)[0]
        return content.strip()[:_USER_TEXT_LIMIT], index
    return "", -1


def _recent_tools(messages: list[Any]) -> list[str]:
    completed = {
        str(_field(message, "tool_call_id"))
        for message in messages
        if _field(message, "type") == "tool"
    }
    seen: set[str] = set()
    tools: list[str] = []
    for message in reversed(messages):
        for call in reversed(_field(message, "tool_calls", []) or []):
            call_id = str(_field(call, "id", ""))
            if not call_id or call_id in seen:
                continue
            seen.add(call_id)
            name = str(_field(call, "name", "инструмент"))[:80]
            tools.append(
                f"{name}: {'завершён' if call_id in completed else 'выполняется'}"
            )
            if len(tools) == _MAX_TOOLS:
                return tools
    return tools


def build_status_context(state: Any) -> tuple[str, list[str]]:
    values = _field(state, "values", {}) or {}
    messages = _field(values, "messages", []) or []
    task, index = _user_task(messages)
    return task, _recent_tools(messages[index + 1 :])


class TelegramStatusService:
    def __init__(self, *, bot_id: Any, owner_id: Any):
        self.bot_id = bot_id
        self.owner_id = owner_id

    async def active_runs(self, client: Any, thread_id: str) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for status in ("running", "pending"):
            runs.extend(
                await client.runs.list(thread_id, limit=100, status=status) or []
            )
        return runs

    async def get_status(self, client: Any, thread_id: str, run_id: str) -> str:
        setup_cache()
        key = f"channel:tg-status:{self.bot_id}:{thread_id}:{run_id}"
        cached = await cache.get(key)
        if isinstance(cached, str):
            return cached
        async with cache.lock(f"{key}:lock", expire=30, wait=True):
            cached = await cache.get(key)
            if isinstance(cached, str):
                return cached
            fallback = "⏳ Агент работает над запросом."
            try:
                state = await client.threads.get_state(thread_id)
                task, tools = build_status_context(state)
                if tools:
                    fallback += " Инструменты: " + "; ".join(tools) + "."
                async with (await get_session_factory())() as session:
                    user = await UserRepository.get_cached_or_db(
                        self.owner_id, session=session
                    )
                if user is None:
                    raise ValueError("Telegram bot owner not found")
                resolver = RuntimeResolver(user)
                runtime = await resolver.get_fast_llm_runtime()
                llm = await runtime.get_llm()
                prompt = (
                    f"Задача пользователя: {task or '[текст недоступен]'}\n"
                    f"Последние инструменты: {'; '.join(tools) if tools else 'нет данных'}"
                )
                response = await llm.with_config(tags=["nostream"]).ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "Кратко, на русском и от третьего лица опиши, что агент делает сейчас. "
                                "Опирайся только на задачу пользователя и состояние инструментов. "
                                "Не утверждай, что работа завершена. Ответь одним предложением."
                            )
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                summary = _message_text(_field(response, "content", "")).strip()
                status_text = summary[:_STATUS_TEXT_LIMIT] or fallback
            except Exception:
                logger.warning(
                    "Could not summarize Telegram run %s", run_id, exc_info=True
                )
                status_text = fallback
            await cache.set(key, status_text, expire=STATUS_TTL_SECONDS)
            return status_text

    async def mark_stopped(self, thread_id: str) -> None:
        setup_cache()
        await cache.set(
            f"channel:tg-stopped:{self.bot_id}:{thread_id}", time.time(), expire=900
        )

    async def clear_stopped(self, thread_id: str) -> None:
        setup_cache()
        await cache.delete(f"channel:tg-stopped:{self.bot_id}:{thread_id}")

    async def was_stopped_since(self, thread_id: str, started_at: float) -> bool:
        setup_cache()
        stopped_at = await cache.get(f"channel:tg-stopped:{self.bot_id}:{thread_id}")
        return isinstance(stopped_at, (int, float)) and stopped_at >= started_at
