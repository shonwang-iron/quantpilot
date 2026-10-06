import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

from quantpilot.compare import load_news, load_periods, timestamp
from quantpilot.feature_strategy import backtest, build_samples, fit_predict

ROOT = Path(__file__).resolve().parents[1] / 'examples'


@unittest.skipUnless(importlib.util.find_spec('sklearn'), '需安裝 requirements-strategy.txt')
class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.news = load_news(ROOT / 'feature_news_demo.csv')
        self.periods = load_periods(ROOT / 'feature_periods_demo.csv')
        self.samples = build_samples(self.news, self.periods)
        self.start = self.periods[6]['_entry']

    def test_ablation_and_actual_regressors(self):
        report = backtest(self.samples, self.start, min_train=3)
        self.assertEqual(len(report['decisions']), 6)
        self.assertEqual(set(report['methods']), {'combined','sentiment_only','words_only','historical_mean','long_every_period','cash'})
        self.assertEqual(report['methods']['cash']['net']['total_return'], 0)
        for decision in report['decisions']:
            self.assertLess(timestamp(decision['train_latest_exit']), timestamp(decision['entry_at']))
            for name in ('combined','sentiment_only','words_only'):
                item=decision['methods'][name]
                self.assertAlmostEqual(item['gross_return']-item['net_return'], 0.002*item['position'])

    def test_future_return_and_news_do_not_affect_past(self):
        before = backtest(self.samples, self.start, min_train=3)
        changed = deepcopy(self.samples)
        changed[-1]['return'] = 0.9
        changed[-1]['text'] = '完全不同未來新聞'
        changed[-1]['features'] = [0]*7
        after = backtest(changed, self.start, min_train=3)
        self.assertEqual(before['decisions'][:-1], after['decisions'][:-1])

    def test_only_training_vocabulary_used(self):
        original, info = fit_predict(self.samples[:3], self.samples[3])
        changed = deepcopy(self.samples[3])
        changed['text'] += ' qzxunknownfuture'
        updated, other = fit_predict(self.samples[:3], changed)
        self.assertEqual(info, other)
        self.assertEqual(original, updated)

    def test_exact_exit_not_available_to_training(self):
        samples=deepcopy(self.samples)
        samples[5]['period']['_exit']=samples[6]['period']['_entry']
        result=backtest(samples,self.start,min_train=3)
        self.assertEqual(result['decisions'][0]['train_periods'],5)

    def test_no_news_forces_cash(self):
        samples=deepcopy(self.samples)
        samples[6]['news_count']=0
        result=backtest(samples,self.start,min_train=3)
        for name in ('combined','sentiment_only','words_only','historical_mean'):
            self.assertEqual(result['decisions'][0]['methods'][name]['position'],0)

    def test_edge_can_prevent_all_forecast_trades(self):
        report=backtest(self.samples,self.start,min_train=3,edge=1)
        for name in ('combined','sentiment_only','words_only','historical_mean'):
            self.assertEqual(report['methods'][name]['net']['round_trips'],0)

    def test_features_use_only_news_before_entry(self):
        rows=deepcopy(self.news)
        rows[-1]['text']='虧損衰退暴跌'
        updated=build_samples(rows,self.periods)
        self.assertEqual(updated[:-1],self.samples[:-1])
        self.assertEqual(self.samples[0]['news_count'],1)
        self.assertEqual(self.samples[0]['features'][0],1)

    def test_insufficient_history_and_invalid_options(self):
        with self.assertRaises(ValueError):
            backtest(self.samples,self.start,min_train=100)
        with self.assertRaises(ValueError):
            backtest(self.samples,self.start,alpha=0)


if __name__ == '__main__':
    unittest.main()
