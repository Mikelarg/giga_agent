import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from qdrant_client import AsyncQdrantClient, models

from giga_agent.channels.telegram.runtime import build_memory_config
from giga_agent.memory import module
from giga_agent.memory.backends.base import MemoryFileDTO
from giga_agent.memory.service import MemoryService


TAG = "tg_chat_10"


def memory(name, content, description=None):
    return MemoryFileDTO(
        id=uuid.uuid4(),
        owner_id=uuid.UUID(int=1),
        tag=TAG,
        path=f"/memories/{TAG}/{name}.md",
        content=content,
        description=description,
        content_hash="hash",
        indexed_hash=None,
        indexed_embedding_id=None,
        updated_at=None,
    )


@pytest.mark.anyio
async def test_scoped_retrieval_groups_chunks_and_reserves_sender_slots(monkeypatch):
    about = memory("ABOUT", "О группе")
    profile = memory("profile_aLiCE", "Профиль участника")
    mentions = memory("games", "Анонс для @ALICE")
    other_profile = memory("profile_alice_smith", "Другой участник")
    guides = [memory(f"guide{i}", "X" * 9000 + f"MATCH{i}") for i in range(6)]
    files = [about, profile, mentions, other_profile, *guides]
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        points = []

        def add(file, distance, *, owner=about.owner_id, tag=TAG, snippet=""):
            points.append(
                models.PointStruct(
                    id=len(points),
                    vector=[1.0, distance],
                    payload={
                        "owner_id": str(owner),
                        "tag": tag,
                        "path": file.path,
                        "chunk_text": snippet,
                    },
                )
            )

        for file in (about, profile, mentions):
            add(file, 0.0)
        for i, file in enumerate(guides):
            for _ in range(5):
                add(file, 0.1 * (i + 1), snippet=f"MATCH{i}")
        # A stronger vector with the same path must not affect the chat ranking.
        add(guides[-1], 0.0, tag="tg_chat_other")
        add(guides[-1], 0.0, owner=uuid.uuid4())
        await client.upsert("test", points=points)
        embedding = SimpleNamespace(aembed_query=AsyncMock(return_value=[1.0, 0.0]))
        svc = MemoryService(
            backend=AsyncMock(),
            owner_id=about.owner_id,
            embedding_runtime=SimpleNamespace(
                get_embeddings=AsyncMock(return_value=embedding)
            ),
            embedding_id=uuid.uuid4(),
        )
        monkeypatch.setattr(
            svc, "_ensure_collection", AsyncMock(return_value=(client, "test", 2))
        )
        monkeypatch.setattr(svc, "list_visible", AsyncMock(return_value=files))
        monkeypatch.setattr(module, "build_memory_service", AsyncMock(return_value=svc))
        state = {
            "messages": [
                HumanMessage("old"),
                AIMessage("answer"),
                ToolMessage("noise", tool_call_id="tool"),
            ]
        }
        prompt = await module.MemoryModule().get_instructions(
            None,
            None,
            state=state,
            config={
                "configurable": {
                    "memory_tags": [TAG],
                    "memory_show_global": False,
                    "memory_username": "Alice",
                    "memory_query": "new question",
                }
            },
        )
        embedding.aembed_query.assert_awaited_once_with("new question")
        assert (
            prompt.index(about.path)
            < prompt.index(profile.path)
            < prompt.index(mentions.path)
            < prompt.index(guides[0].path)
        )
        assert prompt.count("<file ") == 7
        assert guides[-1].path not in prompt and other_profile.path not in prompt
        assert all(f"MATCH{i}" in prompt for i in range(5))
        assert len(prompt) < 12000
    finally:
        await client.close()


@pytest.mark.anyio
@pytest.mark.parametrize("embeddings", [False, True])
async def test_missing_search_keeps_about_profile_and_bounded_fallback(
    monkeypatch, embeddings
):
    # An unavailable embedding provider and legacy unindexed memory both remain usable.
    files = [
        memory("profile_alice", "Профиль"),
        memory("calendar", "Расписание", "календарь"),
        memory("profile_alice_smith", "Похожий ник"),
        memory("other", "пусто" * 10000),
    ]
    svc = SimpleNamespace(
        has_embeddings=embeddings,
        list_visible=AsyncMock(return_value=files),
        search=AsyncMock(side_effect=RuntimeError("offline")),
    )
    monkeypatch.setattr(module, "build_memory_service", AsyncMock(return_value=svc))
    prompt = await module.MemoryModule().get_instructions(
        None,
        None,
        state={"messages": [HumanMessage("календарь"), AIMessage("noise")]},
        config={
            "configurable": {
                "memory_tags": [TAG],
                "memory_show_global": False,
                "memory_username": "alicE",
            }
        },
    )
    assert f"/memories/{TAG}/ABOUT.md" in prompt and "/memories/ABOUT.md" not in prompt
    assert files[0].path in prompt and files[1].path in prompt
    assert files[2].path not in prompt and files[3].path not in prompt
    if embeddings:
        assert svc.search.await_args.args == ("календарь",)
    else:
        svc.search.assert_not_awaited()


@pytest.mark.parametrize(
    ("chat_type", "tag"), [("private", "tg_user_42"), ("group", "tg_chat_10")]
)
def test_button_uses_clicking_user_for_memory(chat_type, tag):
    bot = SimpleNamespace(id=8, username="bot")
    sender = SimpleNamespace(id=42, username="Alice")
    message = SimpleNamespace(
        from_user=bot, chat=SimpleNamespace(id=10, type=chat_type)
    )
    config = build_memory_config(message, "выбор", sender=sender)
    assert config["memory_tags"] == [tag]
    assert config["memory_username"] == "Alice"
    assert config["memory_query"] == "выбор"
