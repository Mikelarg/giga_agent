import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import giga_agent.scheduled.runner as runner
from giga_agent.core.cache import setup_cache
from giga_agent.core.db import Base
from giga_agent.models.channel import ChannelBotRepository
from giga_agent.models.scheduled_task import (
    KIND_CRON,
    STATUS_DONE,
    STATUS_PENDING,
    ScheduledTaskRepository,
)
from giga_agent.models.users import User


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def deliver(
        self, bot, external_chat_id, parts, *, token, external_user_id=None
    ):
        self.calls.append((bot.id, external_chat_id, external_user_id, tuple(parts)))
        return True


class RunnerDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        setup_cache()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.execute(text("PRAGMA foreign_keys=ON"))
            await conn.run_sync(Base.metadata.create_all)

        self.fake_runtime = FakeRuntime()

        async def _fake_get_runtime(channel_type, settings):
            return self.fake_runtime

        self._orig_get_runtime = runner.ChannelRegistry.get_runtime
        runner.ChannelRegistry.get_runtime = staticmethod(_fake_get_runtime)

    async def asyncTearDown(self) -> None:
        runner.ChannelRegistry.get_runtime = self._orig_get_runtime
        await self.engine.dispose()

    async def _user(self, email: str) -> User:
        async with self.session_factory() as session:
            user = User(email=email, hashed_password="h", is_active=True)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def test_explicit_targets_delivered(self) -> None:
        owner = await self._user("o@example.com")
        async with self.session_factory() as session:
            chan = ChannelBotRepository(session)
            bot = await chan.create(user_id=owner.id, channel_type="telegram")
            targets = [{"bot_id": str(bot.id), "external_chat_id": "42"}]
            delivered, failed = await runner._deliver_to_targets(
                chan,
                owner_id=owner.id,
                targets=targets,
                parts=[{"kind": "text", "value": "hi"}],
                token="t",
            )
        self.assertEqual((delivered, failed), (1, 0))
        self.assertEqual(self.fake_runtime.calls[0][1], "42")

    async def test_target_not_owned_counts_as_failed(self) -> None:
        owner = await self._user("o2@example.com")
        other = await self._user("other@example.com")
        async with self.session_factory() as session:
            chan = ChannelBotRepository(session)
            foreign_bot = await chan.create(user_id=other.id, channel_type="telegram")
            targets = [{"bot_id": str(foreign_bot.id), "external_chat_id": "1"}]
            delivered, failed = await runner._deliver_to_targets(
                chan,
                owner_id=owner.id,
                targets=targets,
                parts=[{"kind": "text", "value": "x"}],
                token="t",
            )
        self.assertEqual((delivered, failed), (0, 1))
        self.assertEqual(self.fake_runtime.calls, [])

    async def test_finalize_once_is_terminal(self) -> None:
        owner = await self._user("o3@example.com")
        async with self.session_factory() as session:
            repo = ScheduledTaskRepository(session)
            task = await repo.create(
                owner_id=owner.id,
                name="once",
                prompt="p",
                run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
            await runner._finalize(repo, task, STATUS_DONE)
            refreshed = await repo.get_by_id(task.id)
            self.assertEqual(refreshed.status, STATUS_DONE)

    async def test_finalize_cron_reschedules(self) -> None:
        owner = await self._user("o4@example.com")
        async with self.session_factory() as session:
            repo = ScheduledTaskRepository(session)
            task = await repo.create(
                owner_id=owner.id,
                name="cron",
                prompt="p",
                kind=KIND_CRON,
                cron="*/5 * * * *",
                run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
            await runner._finalize(repo, task, STATUS_DONE)
            refreshed = await repo.get_by_id(task.id)
            self.assertEqual(refreshed.status, STATUS_PENDING)
            # SQLite returns naive datetimes; normalize to UTC for comparison.
            next_run = refreshed.run_at
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            self.assertGreater(next_run, datetime.now(timezone.utc))

    async def test_default_group_is_bound_before_run_and_used_for_delivery(
        self,
    ) -> None:
        owner = await self._user("history-default@example.com")
        async with self.session_factory() as session:
            channels = ChannelBotRepository(session)
            bot = await channels.create(user_id=owner.id, channel_type="telegram")
            await channels.upsert_contact(
                bot_id=bot.id, external_chat_id="-1001", chat_type="supergroup"
            )
            await channels.set_contact_fields_by_external_id(
                bot_id=bot.id,
                external_chat_id="-1001",
                is_approved=True,
                is_default_task_recipient=True,
            )
            task = await ScheduledTaskRepository(session).create(
                owner_id=owner.id, prompt="Сводка"
            )

        with (
            patch.object(
                runner,
                "get_session_factory",
                new=AsyncMock(return_value=self.session_factory),
            ),
            patch.object(runner, "_make_owner_token", new=AsyncMock(return_value="t")),
            patch.object(
                runner,
                "_run_graph",
                new=AsyncMock(
                    return_value={"messages": [{"type": "ai", "content": "Готово"}]}
                ),
            ) as run_graph,
            patch.object(
                runner,
                "_deliver_to_targets",
                new=AsyncMock(return_value=(1, 0)),
            ) as deliver,
        ):
            await runner.execute_due_task(task.id)

        self.assertEqual(
            run_graph.await_args.kwargs["history_target"],
            {"bot_id": str(bot.id), "external_chat_id": "-1001"},
        )
        self.assertEqual(
            deliver.await_args.kwargs["targets"],
            [
                {
                    "bot_id": str(bot.id),
                    "external_chat_id": "-1001",
                    "external_user_id": None,
                }
            ],
        )

    async def test_history_target_rejects_multiple_private_and_unapproved(self) -> None:
        owner = await self._user("history-types@example.com")
        async with self.session_factory() as session:
            channels = ChannelBotRepository(session)
            bot = await channels.create(user_id=owner.id, channel_type="telegram")
            await channels.upsert_contact(
                bot_id=bot.id, external_chat_id="-1001", chat_type="group"
            )
            await channels.set_contact_fields_by_external_id(
                bot_id=bot.id,
                external_chat_id="-1001",
                is_approved=True,
            )
            group = {"bot_id": str(bot.id), "external_chat_id": "-1001"}
            self.assertEqual(
                await runner._history_target(channels, owner.id, [group]), group
            )
            self.assertIsNone(
                await runner._history_target(channels, owner.id, [group, group])
            )
            self.assertIsNone(await runner._history_target(channels, owner.id, []))

            await channels.upsert_contact(
                bot_id=bot.id, external_chat_id="42", chat_type="private"
            )
            await channels.set_contact_fields_by_external_id(
                bot_id=bot.id,
                external_chat_id="42",
                is_approved=True,
            )
            private = {"bot_id": str(bot.id), "external_chat_id": "42"}
            self.assertIsNone(
                await runner._history_target(channels, owner.id, [private])
            )
            await channels.set_contact_fields_by_external_id(
                bot_id=bot.id, external_chat_id="-1001", is_approved=False
            )
            self.assertIsNone(await runner._history_target(channels, owner.id, [group]))

            other_channel = await channels.create(
                user_id=owner.id, channel_type="other"
            )
            await channels.upsert_contact(
                bot_id=other_channel.id,
                external_chat_id="-2001",
                chat_type="group",
            )
            await channels.set_contact_fields_by_external_id(
                bot_id=other_channel.id,
                external_chat_id="-2001",
                is_approved=True,
            )
            self.assertIsNone(
                await runner._history_target(
                    channels,
                    owner.id,
                    [{"bot_id": str(other_channel.id), "external_chat_id": "-2001"}],
                )
            )

    async def test_scheduled_run_passes_history_scope_without_message_tool(
        self,
    ) -> None:
        client = SimpleNamespace(
            threads=SimpleNamespace(
                create=AsyncMock(return_value={"thread_id": "scheduled-1"})
            ),
            runs=SimpleNamespace(wait=AsyncMock(return_value={"messages": []})),
            aclose=AsyncMock(),
        )
        target = {"bot_id": "bot-1", "external_chat_id": "-1001"}
        with patch.object(runner, "get_client", return_value=client):
            await runner._run_graph(
                owner_id=uuid.uuid4(),
                prompt="Сводка",
                task_id=uuid.uuid4(),
                token="t",
                run_timeout=5,
                history_target=target,
            )
        self.assertEqual(
            client.threads.create.await_args.kwargs["metadata"]["history_target"],
            target,
        )
        self.assertEqual(client.runs.wait.await_args.kwargs["input"]["mcp_tools"], [])


if __name__ == "__main__":
    unittest.main()
