import importlib.util
import unittest
import json
import tempfile
from pathlib import Path

from quantpilot.backtest import align_signals, load_bars, load_signals, run_backtest
from quantpilot.compare import timestamp

ROOT = Path(__file__).resolve().parents[1] / 'examples'


@unittest.skipUnless(importlib.util.find_spec('backtesting') and importlib.util.find_spec('pandas'), '需安裝 requirements-backtest.txt')
class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.bars = load_bars(ROOT / 'ohlc_demo.csv')
        self.signals = load_signals(ROOT / 'signals_demo.csv')

    def test_next_open_and_full_holding_period(self):
        bars, _ = align_signals(self.bars, self.signals[:1])
        report, trades, equity = run_backtest(bars, cost_bps=0, hold_bars=2)
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade.EntryBar, 2)
        self.assertEqual(trade.ExitBar, 4)
        self.assertEqual(trade.EntryPrice, bars.Open.iloc[2])
        self.assertEqual(trade.ExitPrice, bars.Open.iloc[4])
        self.assertEqual(len(equity), len(bars))

    def test_commission_is_charged_on_both_sides(self):
        bars, _ = align_signals(self.bars, self.signals[:1])
        _, trades, _ = run_backtest(bars, cost_bps=10)
        trade = trades.iloc[0]
        expected = trade.Size * (trade.EntryPrice + trade.ExitPrice) * 0.001
        self.assertAlmostEqual(trade.Commission, expected)
        self.assertAlmostEqual(trade.PnL, trade.Size * (trade.ExitPrice - trade.EntryPrice) - expected)

    def test_exact_close_signal_waits_another_bar(self):
        signal = {'_time': self.bars.close_at.iloc[1], 'score': 0.9}
        bars, _ = align_signals(self.bars, [signal])
        _, trades, _ = run_backtest(bars)
        self.assertEqual(trades.iloc[0].EntryBar, 3)

    def test_future_signal_cannot_change_previous_trade(self):
        bars, _ = align_signals(self.bars, self.signals[:1])
        _, before, _ = run_backtest(bars)
        changed, _ = align_signals(self.bars, self.signals)
        _, after, _ = run_backtest(changed)
        self.assertEqual(before.iloc[0].EntryBar, after.iloc[0].EntryBar)
        self.assertEqual(before.iloc[0].ExitBar, after.iloc[0].ExitBar)
        self.assertEqual(before.iloc[0].PnL, after.iloc[0].PnL)

    def test_no_signal_and_tail_does_not_leave_open_trade(self):
        bars, _ = align_signals(self.bars, [])
        report, trades, _ = run_backtest(bars)
        self.assertEqual(len(trades), 0)
        self.assertEqual(report['metrics']['Return [%]'], 0)
        late = {'_time': self.bars.close_at.iloc[-1], 'score': 0.9}
        bars, quality = align_signals(self.bars, [late])
        _, trades, _ = run_backtest(bars)
        self.assertEqual(len(trades), 0)
        self.assertEqual(quality['after_last_close'], 1)

    def test_invalid_parameters(self):
        bars, _ = align_signals(self.bars, [])
        with self.assertRaises(ValueError):
            run_backtest(bars, allocation=1)

    def test_prediction_report_preserves_decision_time(self):
        decision_time='2026-01-06T09:01:00+08:00'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'predictions.json'
            path.write_text(json.dumps({'decisions':[{'entry_at':decision_time,'methods':{'combined':{'predicted_return':0.02}}}]}))
            signals=load_signals(path,'combined')
        self.assertEqual(signals[0]['_time'],timestamp(decision_time))
        self.assertEqual(signals[0]['score'],0.02)

    def test_reject_invalid_ohlc(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bars.csv'
            bars=self.bars.reset_index()
            bars.loc[0,'High']=1
            bars.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_bars(path)


if __name__ == '__main__':
    unittest.main()
