"""Resolve the recipients used by one scheduled-task run."""

from __future__ import annotations

from typing import Any

from giga_agent.models.channel import ChannelBotRepository
from giga_agent.models.scheduled_task import ScheduledTask


async def resolve_task_targets(
    channels: ChannelBotRepository, task: ScheduledTask
) -> list[dict[str, Any]]:
    if task.targets:
        return list(task.targets)
    contacts = await channels.list_default_recipients_for_owner(task.owner_id)
    return [
        {
            "bot_id": str(contact.bot_id),
            "external_chat_id": contact.external_chat_id,
            "external_user_id": contact.external_user_id,
        }
        for contact in contacts
    ]
