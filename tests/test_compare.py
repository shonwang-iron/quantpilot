import unittest

from quantpilot.compare import classification, compare, timestamp


def news(text, when, label=''):
    return {'text': text, 'available_at': when, '_time': timestamp(when), 'label': label}


def period(start, end, entry, exit):
    return {'entry_at': start, 'exit_at': end, '_entry': timestamp(start), '_exit': timestamp(end),
            '_open': entry, '_close': exit}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.periods = [period('2026-01-06T09:01:00+08:00', '2026-01-07T09:01:00+08:00', 100, 110),
                        period('2026-01-07T09:01:00+08:00', '2026-01-08T09:01:00+08:00', 110, 99)]

    def test_cost_cash_and_compounding(self):
        rows = [news('獲利成長', '2026-01-06T08:00:00+08:00', 'positive'),
                news('虧損衰退', '2026-01-07T08:00:00+08:00', 'negative')]
        result = compare(rows, self.periods, cost_bps=10)
        self.assertAlmostEqual(result['methods']['lexicon']['net']['total_return'], 0.098)
        self.assertEqual(result['methods']['lexicon']['net']['round_trips'], 1)
        self.assertEqual(result['decisions'][1]['methods']['lexicon']['position'], 0)
        short = compare(rows, self.periods, cost_bps=10, short=True)
        self.assertAlmostEqual(short['methods']['lexicon']['net']['total_return'], 1.098**2 - 1)

    def test_future_news_cannot_change_past_decision(self):
        rows = [news('獲利成長', '2026-01-06T08:00:00+08:00')]
        before = compare(rows, self.periods)
        after = compare(rows + [news('虧損衰退', '2026-01-07T08:00:00+08:00')], self.periods)
        self.assertEqual(before['decisions'][0], after['decisions'][0])

    def test_exact_entry_news_excluded_and_no_news_cash(self):
        result = compare([news('獲利成長', self.periods[0]['entry_at'])], self.periods)
        self.assertEqual(result['decisions'][0]['news_count'], 0)
        self.assertEqual(result['decisions'][0]['methods']['lexicon']['position'], 0)

    def test_future_training_rejected(self):
        with self.assertRaises(ValueError):
            compare([], self.periods, training=[news('獲利', self.periods[0]['entry_at'], 'positive')])

    def test_metrics_and_drawdown(self):
        metrics = classification([('positive', 'positive'), ('negative', 'positive'), ('neutral', 'neutral')])
        self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
        self.assertAlmostEqual(metrics['macro_f1'], (2 / 3 + 0 + 1) / 3)
        from quantpilot.compare import performance
        self.assertAlmostEqual(performance([0.1, -0.2], [1, 1])['max_drawdown'], 0.2)

    def test_timezone_and_threshold_validation(self):
        with self.assertRaises(ValueError):
            timestamp('2026-01-06T09:00:00')
        with self.assertRaises(ValueError):
            compare([], self.periods, threshold=float('nan'))


if __name__ == '__main__':
    unittest.main()
