import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from quantpilot.nlp_sentiment import NLPSentiment


class NLPTests(unittest.TestCase):
    def test_snow_score_conversion_and_tokens(self):
        class Snow:
            def __init__(self, text):
                self.han = text.replace('獲利', '获利')
                self.sentiments = 0.8
                self.words = ['获利', '成长']
        with patch.dict('sys.modules', {'snownlp': SimpleNamespace(SnowNLP=Snow)}):
            result = NLPSentiment().predict('獲利成長')
        self.assertEqual(result['label'], 'positive')
        self.assertAlmostEqual(result['score'], 0.6)
        self.assertIsNone(result['probabilities'])
        self.assertEqual(result['tokens'], ['获利', '成长'])

    def test_language_guard_and_empty(self):
        with self.assertRaises(ValueError):
            NLPSentiment('vader').predict('公司獲利成長')
        with self.assertRaises(ValueError):
            NLPSentiment('snownlp').predict('profit growth')
        self.assertEqual(NLPSentiment().predict('')['status'], 'empty_text')

    @unittest.skipUnless(importlib.util.find_spec('snownlp') and importlib.util.find_spec('vaderSentiment'), '需安裝 requirements-nlp.txt')
    def test_real_packages_and_mixed_language(self):
        model = NLPSentiment()
        results = model.predict_many(['我非常喜歡這個產品', 'This is excellent and wonderful!', 'This is terrible and awful!', '台積電 EPS 成長'])
        self.assertEqual(results[0]['backend'], 'snownlp')
        self.assertTrue(results[0]['tokens'])
        self.assertEqual(results[1]['label'], 'positive')
        self.assertEqual(results[2]['label'], 'negative')
        self.assertEqual(results[3]['status'], 'mixed_language')
        self.assertTrue(all(-1 <= item['score'] <= 1 for item in results))


if __name__ == '__main__':
    unittest.main()
