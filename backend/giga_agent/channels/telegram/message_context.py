"""Helpers for building Telegram message context payloads."""

from __future__ import annotations

from typing import Any

from aiogram import types as tg_types

from giga_agent.channels.telegram.utils import _describe_uploaded_files


def _rich_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_rich_text(item) for item in value)
    if value is None:
        return ""
    return _rich_text(getattr(value, "text", None)) or getattr(
        value, "alternative_text", ""
    )


def _rich_block_text(block: Any) -> str:
    text = _rich_text(getattr(block, "text", None))
    summary = _rich_text(getattr(block, "summary", None))
    caption = getattr(block, "caption", None)
    caption_text = _rich_text(caption)
    nested = [_rich_block_text(item) for item in (getattr(block, "blocks", None) or [])]
    for item in getattr(block, "items", None) or []:
        nested.append(
            " ".join(
                part
                for part in (
                    getattr(item, "label", None),
                    "\n".join(
                        _rich_block_text(child)
                        for child in (getattr(item, "blocks", None) or [])
                    ),
                )
                if part
            )
        )
    for row in getattr(block, "cells", None) or []:
        nested.append(" | ".join(_rich_text(cell.text) for cell in row))
    return "\n".join(part for part in (text, summary, *nested, caption_text) if part)


def get_message_text(message: tg_types.Message | None) -> str:
    if message is None:
        return ""
    text = message.text or message.caption or ""
    if text:
        return text
    rich_message = getattr(message, "rich_message", None)
    if rich_message is None:
        return ""
    return "\n\n".join(
        part for block in rich_message.blocks if (part := _rich_block_text(block))
    )


def _format_message_author(message: tg_types.Message | None) -> tuple[str, str]:
    if message is None:
        return "unknown", "Unknown"

    author = getattr(message, "from_user", None) or getattr(
        message, "sender_chat", None
    )
    username = getattr(author, "username", None)
    username_value = f"@{username}" if username else "unknown"
    name_parts = [
        getattr(author, "first_name", None),
        getattr(author, "last_name", None),
    ]
    full_name = " ".join(part for part in name_parts if part).strip()
    if not full_name:
        full_name = getattr(author, "title", None) or "Unknown"
    return username_value, full_name


def build_reply_kwargs(reply_to_message_id: int | None) -> dict[str, Any]:
    if reply_to_message_id is None:
        return {}
    return {
        "reply_parameters": tg_types.ReplyParameters(message_id=reply_to_message_id)
    }


def build_message_context_payload(
    *,
    label: str,
    message: tg_types.Message | None,
    text: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    username, full_name = _format_message_author(message)
    attachments: list[str] = []
    seen: set[str] = set()
    for file_data in files:
        if not isinstance(file_data, dict):
            continue
        path = file_data.get("path")
        if path and path not in seen:
            attachments.append(path)
            seen.add(path)
    return {
        "label": label,
        "username": username,
        "full_name": full_name,
        "text": text or "[empty]",
        "files": files,
        "attachments": attachments,
    }


def build_message_context(
    *,
    label: str,
    message: tg_types.Message,
    text: str,
    files: list[dict[str, Any]],
    highlight: str | None = None,
) -> str:
    payload = build_message_context_payload(
        label=label,
        message=message,
        text=text,
        files=files,
    )
    lines = [
        f"{payload['label']}:",
        *(["Важно: " + highlight] if highlight else []),
        f"Ник: {payload['username']}",
        f"Имя: {payload['full_name']}",
        "Текст сообщения:",
        payload["text"],
    ]
    if payload["files"]:
        lines.extend(
            [
                "Файлы:",
                _describe_uploaded_files(payload["files"]),
            ]
        )
    return "\n".join(lines)
