import tempfile
import unittest
import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from giga_agent.modules.repl.repl_tools.sentiment import (
    NumpySentimentModel,
    _load_sentiment_model,
    _model_paths,
    predict_sentiments,
)
from giga_agent.modules.repl.repl_tools import sentiment


class NumpySentimentModelTests(unittest.TestCase):
    def test_loads_npz_without_pickle_and_predicts_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sentiment_test.npz"
            np.savez_compressed(
                model_path,
                weights=np.array([[2.0, 0.0], [0.0, 2.0]]),
                bias=np.array([0.0, 0.0]),
                classes=np.array(["negative", "positive"]),
            )

            model = NumpySentimentModel.load(model_path)

        predictions = model.predict_labels(
            np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
        )
        self.assertEqual(predictions.tolist(), ["negative", "positive", "negative"])

    def test_bundled_models_load(self):
        models = {key: _load_sentiment_model(key) for key in _model_paths()}

        self.assertEqual(
            set(models),
            {"Embeddings-2", "EmbeddingsGigaR", "text-embedding-3-small"},
        )
        self.assertTrue(all(model.weights.shape[0] == 3 for model in models.values()))
        self.assertTrue(
            all(
                model.classes.tolist() == ["negative", "neutral", "positive"]
                for model in models.values()
            )
        )

    def test_rejects_wrong_embedding_dimension(self):
        model = NumpySentimentModel(
            weights=np.zeros((2, 3)),
            bias=np.zeros(2),
            classes=np.array(["negative", "positive"]),
        )

        with self.assertRaisesRegex(ValueError, "Embedding dimensions"):
            model.predict_labels(np.zeros((1, 2)))

    def test_import_does_not_start_preloading(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
from unittest.mock import patch
import numpy
with patch('numpy.load', side_effect=AssertionError('unexpected model load')), \
     patch('threading.Thread.start', side_effect=AssertionError('unexpected thread')):
    import giga_agent.modules.repl.repl_tools.availability
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_only_requested_model_is_loaded_and_cached(self):
        model = object()
        paths = {"selected": Path("selected.npz"), "other": Path("broken.npz")}
        with (
            patch.dict(sentiment._SENTIMENT_MODELS, {}, clear=True),
            patch.object(sentiment, "_model_paths", return_value=paths),
            patch.object(NumpySentimentModel, "load", return_value=model) as load,
        ):
            self.assertIs(_load_sentiment_model("selected"), model)
            self.assertIs(_load_sentiment_model("selected"), model)
            load.assert_called_once_with(paths["selected"])
            self.assertEqual(set(sentiment._SENTIMENT_MODELS), {"selected"})

    def test_missing_embeddings_do_not_load_classifier(self):
        with (
            patch.object(
                sentiment,
                "_resolve_user_embeddings",
                new=AsyncMock(side_effect=ValueError("No embedding configured")),
            ),
            patch.object(sentiment, "_load_sentiment_model") as load,
        ):
            with self.assertRaisesRegex(ValueError, "No embedding configured"):
                asyncio.run(predict_sentiments(texts=["test"], tool_runtime=object()))
            load.assert_not_called()
