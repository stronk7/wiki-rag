#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.search.util tests."""

import unittest

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from wiki_rag.search.util import (
    RagState,
    load_prompts_for_rag_from_local,
    route_after_rewrite,
)


def _make_context(
    enable_hyde: bool = False,
    contextualisation_model: str = "",
    hyde_model: str | None = None,
    hyde_passages: int = 1,
) -> SimpleNamespace:
    """Build a minimal runtime-like namespace for routing tests."""
    # hyde_model falls back to contextualisation_model, mirroring build_context_schema().
    resolved_hyde_model = hyde_model or contextualisation_model or ""
    return SimpleNamespace(context={
        "hyde_enabled": enable_hyde,
        "contextualisation_model": contextualisation_model,
        "hyde_model": resolved_hyde_model,
        "hyde_passages": hyde_passages,
    })


def _make_state(**kwargs) -> RagState:
    """Build a minimal RagState for routing tests."""
    defaults: dict = {
        "history": [],
        "question": "What is a Moodle quiz?",
        "hyde_texts": [],
        "vector_search": [],
        "context": [],
        "sources": [],
    }
    defaults.update(kwargs)
    return defaults  # type: ignore[return-value]


class TestRouteAfterRewrite(unittest.TestCase):

    def test_route_after_rewrite_chitchat(self):
        """Returns 'chitchat' when answer is already set, regardless of hyde settings."""
        state = _make_state(answer="Hello there!")
        runtime = _make_context(enable_hyde=True, contextualisation_model="gpt-4o")
        self.assertEqual("chitchat", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_enabled(self):
        """Returns 'hyde_rewrite' when hyde_enabled is True and a model is available."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, contextualisation_model="gpt-4o-mini")
        self.assertEqual("hyde_rewrite", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_enabled_with_dedicated_model(self):
        """Returns 'hyde_rewrite' when hyde_enabled is True and hyde_model is set directly."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, hyde_model="hyde-specific-model")
        self.assertEqual("hyde_rewrite", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_disabled(self):
        """Returns 'retrieve' when hyde_enabled is False, even with a model set."""
        state = _make_state()
        runtime = _make_context(enable_hyde=False, contextualisation_model="gpt-4o-mini")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_hyde_no_model(self):
        """Returns 'retrieve' when hyde_enabled is True but no model is available."""
        state = _make_state()
        runtime = _make_context(enable_hyde=True, contextualisation_model="")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]

    def test_route_after_rewrite_default_retrieve(self):
        """Returns 'retrieve' when hyde is disabled and no contextualisation model."""
        state = _make_state()
        runtime = _make_context(enable_hyde=False, contextualisation_model="")
        self.assertEqual("retrieve", route_after_rewrite(state, runtime))  # type: ignore[arg-type]


class TestLoadPromptsForRagFromLocalHyde(unittest.TestCase):

    def test_hyde_prompt_input_variables(self):
        """The wiki-rag-hyde prompt exposes the expected input variables."""
        prompt = load_prompts_for_rag_from_local("wiki-rag-hyde")
        variables = set(prompt.input_variables)
        self.assertIn("question", variables)
        self.assertIn("task_def", variables)
        self.assertIn("kb_name", variables)
        self.assertIn("product", variables)

    def test_hyde_prompt_message_count(self):
        """The wiki-rag-hyde prompt has system + history placeholder + user messages."""
        from langchain_core.prompts import MessagesPlaceholder
        prompt = load_prompts_for_rag_from_local("wiki-rag-hyde")
        self.assertEqual(3, len(prompt.messages))
        self.assertIsInstance(prompt.messages[1], MessagesPlaceholder)


class TestRetrieveNode(unittest.IsolatedAsyncioTestCase):

    def _make_runtime(self) -> SimpleNamespace:
        return SimpleNamespace(context={
            "collection_name": "test_collection",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 512,
            "embedding_api_base": "https://api.example.com/v1",
            "embedding_api_key": "test-embed-key",  # pragma: allowlist secret
        })

    async def test_retrieve_without_hyde_uses_question_for_both(self):
        """Without HyDE, the question is passed as queries[0]; sparse_query is None."""
        from wiki_rag.search.util import retrieve

        state = _make_state(question="What is a quiz?", hyde_texts=[])
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[{"id": "1"}])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=["What is a quiz?"],
            sparse_query=None,
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )

    async def test_retrieve_with_hyde_uses_passages_for_dense_question_for_sparse(self):
        """With HyDE, only passages go to dense queries; original question is sparse_query."""
        from wiki_rag.search.util import retrieve

        passages = ["A quiz is a formative assessment tool.", "Teachers use quizzes to test recall."]
        state = _make_state(question="What is a quiz?", hyde_texts=passages)
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[{"id": "1"}])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=passages,
            sparse_query="What is a quiz?",
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )

    async def test_retrieve_with_empty_hyde_texts_falls_back_to_question(self):
        """An empty hyde_texts list is treated the same as no HyDE (falsy check)."""
        from wiki_rag.search.util import retrieve

        state = _make_state(question="What is a quiz?", hyde_texts=[])
        runtime = self._make_runtime()

        mock_store = MagicMock()
        mock_store.retrieve = MagicMock(return_value=[])

        with patch("wiki_rag.search.util.vector") as mock_vector:
            mock_vector.store = mock_store
            await retrieve(state, runtime)  # type: ignore[arg-type]

        mock_store.retrieve.assert_called_once_with(
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=512,
            queries=["What is a quiz?"],
            sparse_query=None,
            embedding_api_base="https://api.example.com/v1",
            embedding_api_key="test-embed-key",  # pragma: allowlist secret
        )


class TestHydeNode(unittest.IsolatedAsyncioTestCase):

    async def test_hyde_node_single_passage_returns_texts(self):
        """The hyde_rewrite node returns hyde_texts with the generated passage."""
        from wiki_rag.search.util import hyde_rewrite

        fake_passage = "Moodle quizzes allow teachers to build question banks."

        mock_llm_response = MagicMock()
        mock_llm_response.content = fake_passage

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        mock_prompt = AsyncMock()
        mock_prompt.ainvoke = AsyncMock(return_value=MagicMock())

        state = _make_state()
        runtime = _make_context(
            enable_hyde=True,
            contextualisation_model="gpt-4o-mini",
            hyde_passages=1,
        )
        runtime.context.update({
            "prompt_name": "wiki-rag",
            "product": "Moodle",
            "task_def": "Moodle user documentation",
            "kb_name": "Moodle Docs",
            "hyde_api_base": "https://api.example.com/v1",
            "hyde_api_key": "test-hyde-key",  # pragma: allowlist secret
        })

        with (
            patch("wiki_rag.search.util.load_prompts_for_rag", return_value=mock_prompt),
            patch("wiki_rag.search.util.ChatOpenAI", MagicMock(return_value=mock_llm)),
        ):
            result = await hyde_rewrite(state, runtime)  # type: ignore[arg-type]

        self.assertIn("hyde_texts", result)
        self.assertEqual([fake_passage], result["hyde_texts"])

    async def test_hyde_node_multiple_passages_returns_all_texts(self):
        """The hyde_rewrite node returns hyde_texts with all generated passages."""
        from wiki_rag.search.util import hyde_rewrite

        fake_passage = "Some passage."

        mock_llm_response = MagicMock()
        mock_llm_response.content = fake_passage

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        mock_prompt = AsyncMock()
        mock_prompt.ainvoke = AsyncMock(return_value=MagicMock())

        state = _make_state()
        runtime = _make_context(
            enable_hyde=True,
            contextualisation_model="gpt-4o-mini",
            hyde_passages=3,
        )
        runtime.context.update({
            "prompt_name": "wiki-rag",
            "product": "Moodle",
            "task_def": "Moodle user documentation",
            "kb_name": "Moodle Docs",
            "hyde_api_base": "https://api.example.com/v1",
            "hyde_api_key": "test-hyde-key",  # pragma: allowlist secret
        })

        with (
            patch("wiki_rag.search.util.load_prompts_for_rag", return_value=mock_prompt),
            patch("wiki_rag.search.util.ChatOpenAI", MagicMock(return_value=mock_llm)),
        ):
            result = await hyde_rewrite(state, runtime)  # type: ignore[arg-type]

        self.assertIn("hyde_texts", result)
        self.assertEqual(3, len(result["hyde_texts"]))
        self.assertTrue(all(t == fake_passage for t in result["hyde_texts"]))


if __name__ == "__main__":
    unittest.main()
