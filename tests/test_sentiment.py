import unittest

from quantpilot.sentiment import NaiveBayes, cluster_news, lexicon_sentiment


class SentimentTests(unittest.TestCase):
    def test_lexicon_polarity_and_negation(self):
        self.assertEqual(lexicon_sentiment("獲利成長")['label'], 'positive')
        self.assertEqual(lexicon_sentiment("虧損惡化")['label'], 'negative')
        self.assertEqual(lexicon_sentiment("未成長")['label'], 'negative')
        self.assertEqual(lexicon_sentiment("公司發布公告")['label'], 'neutral')
        self.assertEqual(lexicon_sentiment("profitability")['label'], 'neutral')

    def test_supervised_predictions(self):
        model = NaiveBayes().fit([
            {"text": "獲利成長 profit growth", "label": "positive"},
            {"text": "虧損衰退 loss decline", "label": "negative"},
            {"text": "公司公告 meeting notice", "label": "neutral"},
        ])
        for text, label in (("profit growth", "positive"), ("loss decline", "negative"), ("meeting notice", "neutral")):
            result = model.predict(text)
            self.assertEqual(result['label'], label)
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
        self.assertEqual(model.predict("unknown")['status'], 'no_known_tokens')

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayes().fit([{"text": "成長", "label": "positive"}])

    def test_clustering_groups_identical_news(self):
        texts = ["profit growth", "profit growth", "loss decline", "meeting notice"]
        result = cluster_news(texts)
        self.assertEqual(result[0]['cluster'], result[1]['cluster'])
        self.assertNotEqual(result[0]['cluster'], result[2]['cluster'])
        self.assertEqual(result[0]['interpreted_label'], 'positive')
        self.assertEqual(result[2]['interpreted_label'], 'negative')
        self.assertEqual(cluster_news([]), [])


if __name__ == '__main__':
    unittest.main()
