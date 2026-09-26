"""Message handlers for Telegram runtime."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from aiogram import types as tg_types

from giga_agent.channels.telegram.constants import (
    GROUP_CHAT_TYPES,
    RUN_WAIT_TIMEOUT_SECONDS,
)
from giga_agent.channels.telegram.message_context import (
    build_message_context,
    build_reply_kwargs,
    get_message_text,
)
from giga_agent.channels.telegram.message_tool import build_telegram_message_tool_schema
from giga_agent.channels.telegram.runtime import build_memory_tags
from giga_agent.channels.telegram.services.access import TelegramAccessService
from giga_agent.channels.telegram.services.media import TelegramMediaService
from giga_agent.channels.telegram.services.message_tool_runtime import (
    TelegramMessageToolRuntime,
)
from giga_agent.channels.telegram.services.threads import TelegramThreadService
from giga_agent.channels.telegram.services.status import TelegramStatusService
from giga_agent.core.db import get_session_factory
from giga_agent.core.logging import get_logger
from giga_agent.models.channel import ChannelBot, ChannelBotRepository

logger = get_logger(__name__)


class TelegramMessageHandlers:
    def __init__(
        self,
        *,
        bot_row: ChannelBot,
        access_service: TelegramAccessService,
        thread_service: TelegramThreadService,
        media_service: TelegramMediaService,
        message_tool_runtime: TelegramMessageToolRuntime,
        status_service: TelegramStatusService,
    ):
        self.bot_row = bot_row
        self.access_service = access_service
        self.thread_service = thread_service
        self.media_service = media_service
        self.message_tool_runtime = message_tool_runtime
        self.status_service = status_service

    async def handle_start(self, message: tg_types.Message) -> None:
        if not await self.access_service.ensure_supported_chat(message):
            return
        await self.access_service.register_contact(message)
        await message.answer(
            "Привет! Я GigaAgent — универсальный AI-агент.\n\n"
            "Просто напишите мне сообщение, и я отвечу.\n"
            "Можете отправлять фото, документы и голосовые.\n"
            "/new — начать новый диалог (сбросить контекст)"
            "\n/status — узнать, что я делаю сейчас"
            "\n/stop — остановить текущую работу"
        )

    async def _command_thread_id(self, message: tg_types.Message) -> str | None:
        await self.access_service.register_contact(message)
        async with (await get_session_factory())() as session:
            repo = ChannelBotRepository(session)
            contact = await repo.get_contact(self.bot_row.id, str(message.chat.id))
            if contact is None or not contact.is_approved:
                await self._answer_command(
                    message,
                    "⏳ Ваш контакт ожидает подтверждения. "
                    "Владелец бота должен одобрить вас в настройках.",
                )
                return None
            thread = await repo.get_thread(
                self.bot_row.id,
                str(message.chat.id),
                self.thread_service.resolve_external_user_id(message),
            )
            return thread.langgraph_thread_id if thread else ""

    async def _answer_command(self, message: tg_types.Message, text: str) -> None:
        await message.answer(text, **build_reply_kwargs(message.message_id))

    async def handle_status(self, message: tg_types.Message) -> None:
        if not await self.access_service.ensure_supported_chat(
            message, reply_to_message_id=message.message_id
        ):
            return
        thread_id = await self._command_thread_id(message)
        if thread_id is None:
            return
        if not thread_id:
            await self._answer_command(message, "Сейчас агент не выполняет задачу.")
            return
        client = self.thread_service.create_client(self.thread_service.create_token())
        try:
            runs = await self.status_service.active_runs(client, thread_id)
            if not runs:
                await self._answer_command(message, "Сейчас агент не выполняет задачу.")
                return
            if await self.message_tool_runtime.get_pending_message_tool_calls(
                client, thread_id
            ):
                await self._answer_command(
                    message, "Сейчас агент ожидает вашего ответа."
                )
                return
            run_id = str(runs[0].get("run_id") or "")
            if not run_id:
                await self._answer_command(message, "⏳ Агент работает над запросом.")
                return
            await self._answer_command(
                message, await self.status_service.get_status(client, thread_id, run_id)
            )
        except Exception:
            logger.warning("Failed to fetch Telegram status", exc_info=True)
            await self._answer_command(
                message, "⚠️ Не удалось получить статус. Попробуйте позже."
            )
        finally:
            await client.aclose()

    async def handle_stop(self, message: tg_types.Message) -> None:
        if not await self.access_service.ensure_supported_chat(
            message, reply_to_message_id=message.message_id
        ):
            return
        thread_id = await self._command_thread_id(message)
        if thread_id is None:
            return
        if not thread_id:
            await self._answer_command(
                message, "Сейчас нет активного запуска для остановки."
            )
            return
        client = self.thread_service.create_client(self.thread_service.create_token())
        try:
            runs = await self.status_service.active_runs(client, thread_id)
            if not runs:
                await self._answer_command(
                    message, "Сейчас нет активного запуска для остановки."
                )
                return
            if await self.message_tool_runtime.get_pending_message_tool_calls(
                client, thread_id
            ):
                await self._answer_command(
                    message, "Сейчас нет активного запуска для остановки."
                )
                return
            await self.status_service.mark_stopped(thread_id)
            count = await self.thread_service.stop_thread_runs(client, thread_id)
            if count:
                await self._answer_command(
                    message, "🛑 Работа остановлена. Можете написать новую задачу."
                )
            else:
                await self.status_service.clear_stopped(thread_id)
                await self._answer_command(
                    message, "Сейчас нет активного запуска для остановки."
                )
        except Exception:
            await self.status_service.clear_stopped(thread_id)
            logger.warning("Failed to stop Telegram run", exc_info=True)
            await self._answer_command(
                message, "⚠️ Не удалось остановить работу. Попробуйте ещё раз."
            )
        finally:
            await client.aclose()

    async def handle_new(self, message: tg_types.Message) -> None:
        if not await self.access_service.ensure_supported_chat(message):
            return
        session_factory = await get_session_factory()
        external_user_id = self.thread_service.resolve_external_user_id(message)
        async with session_factory() as session:
            repo = ChannelBotRepository(session)
            await self.thread_service.reset_thread(
                repo,
                message.chat.id,
                external_user_id,
            )
        await message.answer(
            "🔄 Контекст сброшен. Начнём новый диалог!",
            reply_markup=tg_types.ReplyKeyboardRemove(),
        )

    async def handle_message(
        self,
        message: tg_types.Message,
        *,
        force_process: bool = False,
        reply_to_message_id: int | None = None,
        contact_message: tg_types.Message | None = None,
        text_override: str | None = None,
        extra_file_messages: list[tg_types.Message] | None = None,
    ) -> None:
        if not await self.access_service.ensure_supported_chat(message):
            return
        if not force_process and not self.access_service.should_process_message(
            message
        ):
            return
        if contact_message is None:
            contact_message = message
        chat_id = message.chat.id
        if reply_to_message_id is None and message.chat.type in GROUP_CHAT_TYPES:
            reply_to_message_id = message.message_id
        raw_text = get_message_text(message)
        text = (
            text_override
            if text_override is not None
            else self.access_service.strip_bot_mentions(raw_text)
        )
        user_id = self.bot_row.user_id
        reply_kwargs = build_reply_kwargs(reply_to_message_id)

        logger.info("Telegram message from chat %s: %s", chat_id, text[:100])
        request_start = datetime.now(timezone.utc)
        request_start_epoch = time.time()

        client = None
        thread_id: str | None = None
        try:
            session_factory = await get_session_factory()

            await self.access_service.register_contact(contact_message)
            async with session_factory() as session:
                repo = ChannelBotRepository(session)
                contact = await repo.get_contact(self.bot_row.id, str(chat_id))
                if contact is None or not contact.is_approved:
                    await message.answer(
                        "⏳ Ваш контакт ожидает подтверждения. "
                        "Владелец бота должен одобрить вас в настройках.",
                        **reply_kwargs,
                    )
                    return

            token = self.thread_service.create_token()
            client = self.thread_service.create_client(token)
            external_user_id = self.thread_service.resolve_external_user_id(message)

            async with session_factory() as session:
                repo = ChannelBotRepository(session)
                thread_id = await self.thread_service.get_or_create_thread(
                    client,
                    repo,
                    chat_id,
                    external_user_id,
                )

            run_timeout = RUN_WAIT_TIMEOUT_SECONDS
            pending_message_tools = (
                await self.message_tool_runtime.get_pending_message_tool_calls(
                    client,
                    thread_id,
                )
            )
            if pending_message_tools:
                result = await self.message_tool_runtime.resume_pending_message_tool(
                    message=message,
                    client=client,
                    thread_id=thread_id,
                    token=token,
                    pending_tool_calls=pending_message_tools,
                    run_timeout=run_timeout,
                    reply_to_message_id=reply_to_message_id,
                    extra_file_messages=extra_file_messages,
                )
                if result is None:
                    return
            elif await self.message_tool_runtime.has_active_run(client, thread_id):
                await message.answer(
                    "⏳ Бот ещё обрабатывает предыдущее сообщение. "
                    "Напишите /status, чтобы узнать, что он делает, "
                    "или /stop, чтобы остановить. Дождитесь завершения и повторите запрос.",
                    **reply_kwargs,
                )
                return
            else:
                await message.chat.do("typing")
                reply_message = getattr(message, "reply_to_message", None)
                reply_text = ""
                reply_file_data: list[dict[str, Any]] = []
                if reply_message is not None:
                    reply_text = get_message_text(reply_message)
                    reply_file_data = await self.media_service.collect_incoming_files(
                        reply_message,
                        token,
                        thread_id,
                    )

                file_data = await self.media_service.collect_incoming_files(
                    message,
                    token,
                    thread_id,
                )
                # Album/gallery parts beyond the primary message arrive as
                # separate updates; fold their files into this single run.
                for extra in extra_file_messages or []:
                    extra_files = await self.media_service.collect_incoming_files(
                        extra,
                        token,
                        thread_id,
                    )
                    file_data.extend(extra_files)
                all_file_data = [*reply_file_data, *file_data]
                if all_file_data:
                    logger.info(
                        "Files for agent: %s",
                        [item["path"] for item in all_file_data],
                    )

                content_parts = [
                    build_message_context(
                        label="Входящее сообщение",
                        message=message,
                        text=text,
                        files=file_data,
                    )
                ]
                if reply_message is not None:
                    reply_label = "Прикреплено сообщение"
                    reply_highlight = None
                    if self.access_service.is_current_bot_message(reply_message):
                        reply_label = "Прикреплено твое сообщение"
                        reply_highlight = (
                            "Это твое предыдущее сообщение. "
                            "Учитывай его как свой прошлый ответ в диалоге."
                        )
                    content_parts.append(
                        build_message_context(
                            label=reply_label,
                            message=reply_message,
                            text=reply_text,
                            files=reply_file_data,
                            highlight=reply_highlight,
                        )
                    )
                content = "\n\n".join(content_parts)
                content += (
                    "\n\n[system: Ответ будет отправлен в Telegram. "
                    "Если ты выполняешь долгую операцию, например вызываешь субагентов, "
                    "то вызови message тул с уведомлением о том, что будешь делать с "
                    "except_response=False"
                    "Активно планируй и следуй своему плану! Всегда перед вызовом тулов "
                    "размышляй над задачей и пиши размышления в тег <thinking>"
                    "Действуй по простым шагам!"
                    "Следующий шаг: "
                )
                human_msg: dict[str, Any] = {"role": "human", "content": content}
                if all_file_data:
                    human_msg["additional_kwargs"] = {
                        "user_input": content,
                        "files": all_file_data,
                    }

                run_input: dict[str, Any] = {
                    "messages": [human_msg],
                    "mcp_tools": [build_telegram_message_tool_schema()],
                }
                collections_payload = (
                    await self.thread_service.load_collections_payload()
                )
                if collections_payload:
                    run_input["collections"] = collections_payload

                result = await asyncio.wait_for(
                    client.runs.wait(
                        thread_id=thread_id,
                        assistant_id=self.thread_service.assistant_id,
                        input=run_input,
                        config={
                            "configurable": {
                                "memory_disabled": False,
                                "memory_tags": build_memory_tags(message),
                                "memory_show_global": False,
                                # Migrate pre-existing threads to autonomous mode:
                                # before_agent syncs this into thread metadata.
                                "auto_approve": True,
                            },
                        },
                    ),
                    timeout=run_timeout,
                )
                result = await self.message_tool_runtime.continue_run_until_ready(
                    message=message,
                    client=client,
                    thread_id=thread_id,
                    token=token,
                    result=result,
                    run_timeout=run_timeout,
                    reply_to_message_id=reply_to_message_id,
                )
                if result is None:
                    return

            # A cancelled run can resolve with an empty result. Check the stop
            # marker before treating that result as a broken thread.
            if await self.status_service.was_stopped_since(
                thread_id, request_start_epoch
            ):
                return

            if not isinstance(result, dict) or not result.get("messages"):
                logger.warning(
                    "Empty result for chat %s: %s",
                    chat_id,
                    str(result)[:500],
                )
                async with session_factory() as session:
                    repo = ChannelBotRepository(session)
                    await self.thread_service.reset_thread(
                        repo,
                        chat_id,
                        external_user_id,
                    )
                await message.answer(
                    "⚠️ Агент не вернул ответ. Попробуйте ещё раз.",
                    **reply_kwargs,
                )
                return

            await self.media_service.send_run_result(
                message=message,
                token=token,
                result=result,
                request_start=request_start,
                reply_to_message_id=reply_to_message_id,
            )

        except asyncio.TimeoutError:
            if thread_id and await self.status_service.was_stopped_since(
                thread_id, request_start_epoch
            ):
                return
            logger.warning(
                "Timeout handling Telegram message for user %s (chat %s)",
                user_id,
                chat_id,
            )
            try:
                async with (await get_session_factory())() as session:
                    repo = ChannelBotRepository(session)
                    await self.thread_service.reset_thread(
                        repo,
                        chat_id,
                        self.thread_service.resolve_external_user_id(message),
                        stop_thread=False,
                    )
            except Exception:
                pass
            try:
                await message.answer(
                    "⏱ Время ожидания ответа истекло. Попробуйте ещё раз.",
                    **reply_kwargs,
                )
            except Exception:
                pass
        except Exception as exc:
            if thread_id and await self.status_service.was_stopped_since(
                thread_id, request_start_epoch
            ):
                return
            logger.exception("Error handling Telegram message for user %s", user_id)
            error_str = str(exc)
            if "UserInterrupt" in error_str:
                return
            if "tool_call" in error_str or "function call" in error_str:
                try:
                    async with (await get_session_factory())() as session:
                        repo = ChannelBotRepository(session)
                        await self.thread_service.reset_thread(
                            repo,
                            chat_id,
                            self.thread_service.resolve_external_user_id(message),
                        )
                except Exception:
                    pass
                try:
                    await message.answer(
                        "⚠️ Контекст повреждён, сброшен. Повторите сообщение.",
                        **reply_kwargs,
                    )
                except Exception:
                    pass
            else:
                try:
                    await message.answer(
                        "⚠️ Произошла ошибка при обработке сообщения.",
                        **reply_kwargs,
                    )
                except Exception:
                    pass
        finally:
            if client is not None:
                await client.aclose()


def _has_supported_attachment(message: tg_types.Message) -> bool:
    return bool(
        message.photo
        or message.document
        or message.voice
        or message.audio
        or message.video
        or message.video_note
        or message.sticker
    )
