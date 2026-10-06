import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from quantpilot.deep_sentiment import DeepSentiment


class DeepTests(unittest.TestCase):
    def make_pipeline(self):
        classifier = Mock()
        classifier.model.config = SimpleNamespace(id2label={0: 'positive', 1: 'negative', 2: 'neutral'}, max_position_embeddings=512)
        classifier.tokenizer.model_max_length = 512
        classifier.tokenizer.side_effect = lambda text, **kwargs: {'input_ids': list(range(len(text) + 2))}
        classifier.side_effect = lambda texts, **kwargs: [[{'label': 'negative', 'score': 0.1},
            {'label': 'neutral', 'score': 0.2}, {'label': 'positive', 'score': 0.7}] for text in texts]
        return classifier

    def test_batching_labels_and_truncation(self):
        classifier = self.make_pipeline()
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(pipeline=Mock(return_value=classifier))}):
            model = DeepSentiment(batch_size=2, max_length=8)
            results = model.predict_many(['成長', '長' * 20, 'profit'])
        self.assertEqual(len(results), 3)
        self.assertEqual(classifier.call_count, 2)
        self.assertEqual(results[0]['label'], 'positive')
        self.assertAlmostEqual(results[0]['score'], 0.6)
        self.assertEqual(results[1]['status'], 'truncated')
        self.assertEqual(model.predict_many([]), [])

    def test_reject_ambiguous_labels(self):
        classifier = self.make_pipeline()
        classifier.model.config.id2label = {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(pipeline=Mock(return_value=classifier))}):
            with self.assertRaises(ValueError):
                DeepSentiment()

    def test_reject_invalid_probabilities(self):
        classifier = self.make_pipeline()
        classifier.side_effect = None
        classifier.return_value = [[{'label': 'positive', 'score': 1.0}]]
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(pipeline=Mock(return_value=classifier))}):
            model = DeepSentiment()
            with self.assertRaises(ValueError):
                model.predict_many(['news'])

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            DeepSentiment(batch_size=0)


if __name__ == '__main__':
    unittest.main()
