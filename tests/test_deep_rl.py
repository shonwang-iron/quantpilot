import importlib.util
from pathlib import Path
import tempfile
import unittest

from quantpilot.compare import timestamp


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy', 'pandas')),
                     'requires requirements-rl.txt')
class DeepRLTests(unittest.TestCase):
    def setUp(self):
        from quantpilot.backtest import load_bars
        from quantpilot.deep_rl import TradingMarket
        self.bars = load_bars(Path(__file__).resolve().parents[1] / 'examples/ohlc_demo.csv')
        self.env = TradingMarket(self.bars, 4, 8, window=3, cash=1000, cost_bps=10)

    def test_state_is_sigmoid_past_differences(self):
        import math
        import numpy as np
        state = self.env.observation()
        np.testing.assert_allclose(state, [1 / (1 + math.exp(-1))] * 3)
        self.bars.loc[self.bars.index[4:], 'Close'] = 500
        np.testing.assert_array_equal(state, self.env.observation())

    def test_only_sell_rewards_net_pnl(self):
        from quantpilot.deep_rl import BUY, HOLD, SELL
        self.assertEqual(self.env.step(BUY)[1], 0)
        self.assertEqual(self.env.step(HOLD)[1], 0)
        _, reward, _, _ = self.env.step(SELL)
        expected = 106 * .999 - 104 * 1.001
        self.assertAlmostEqual(reward, expected)
        self.env.step(HOLD)
        self.assertAlmostEqual(self.env.metrics()['final_cash'], 1000 + expected)
        self.assertTrue(all(row['observation_at'] < row['execution_at'] for row in self.env.records))

    def test_masks_and_explicit_terminal_sale(self):
        from quantpilot.deep_rl import BUY, HOLD, SELL
        with self.assertRaises(ValueError):
            self.env.step(SELL)
        self.env.step(BUY)
        with self.assertRaises(ValueError):
            self.env.step(BUY)
        self.env.step(HOLD)
        self.env.step(HOLD)
        self.assertEqual(self.env.mask().tolist(), [False, False, True])
        with self.assertRaises(ValueError):
            self.env.step(HOLD)
        _, reward, done, _ = self.env.step(SELL)
        self.assertTrue(done)
        self.assertTrue(self.env.records[-1]['forced_exit'])
        self.assertAlmostEqual(reward, self.env.metrics()['realized_pnl'])

    def test_opening_gap_rejects_unaffordable_order(self):
        from quantpilot.deep_rl import TradingMarket, BUY
        self.bars.loc[self.bars.index[4], 'Open'] = 110
        env = TradingMarket(self.bars, 4, 8, window=3, cash=105)
        self.assertTrue(env.mask()[BUY])
        self.assertEqual(env.step(BUY)[1], 0)
        self.assertFalse(env.records[-1]['filled'])
        self.assertEqual(env.cash, 105)

    def test_loss_is_negative_realized_reward(self):
        from quantpilot.deep_rl import BUY, HOLD, SELL
        self.env.step(BUY)
        self.env.step(HOLD)
        self.env.step(HOLD)
        self.bars.loc[self.bars.index[7], 'Open'] = 50
        _, reward, _, _ = self.env.step(SELL)
        self.assertLess(reward, 0)
        self.assertAlmostEqual(reward, 50 * .999 - 104 * 1.001)
        self.assertAlmostEqual(self.env.cash - 1000, reward)

    def test_extreme_sigmoid_differences_are_finite(self):
        import numpy as np
        self.bars.loc[self.bars.index[0:4], 'Close'] = [1, 1000000, 1, 1000000]
        self.assertTrue(np.isfinite(self.env.observation()).all())
        np.testing.assert_array_equal(self.env.observation(), [1, 0, 1])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_real_gradient_training_and_checkpoint(self):
        import torch
        from quantpilot.deep_rl import train, network
        model, diagnostics = train(self.env, episodes=8, batch_size=4)
        self.assertGreater(diagnostics['gradient_updates'], 0)
        self.assertTrue(diagnostics['weights_changed'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            torch.save(model.state_dict(), path)
            restored = network(3, 32)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(model(torch.tensor(self.env.observation())),
                                        restored(torch.tensor(self.env.observation()))))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_test_prices_do_not_affect_model_selection(self):
        import torch
        from quantpilot.deep_rl import research
        kwargs = dict(window=3, episodes=4, batch_size=4, learning_rates=(.001,), gammas=(.9,))
        dates = timestamp('2026-01-15T00:00:00+08:00'), timestamp('2026-01-21T00:00:00+08:00')
        report, model = research(self.bars, *dates, **kwargs)
        changed = self.bars.copy()
        changed.loc[changed.index[12:], ['Open', 'High', 'Low', 'Close']] *= 2
        second, other = research(changed, *dates, **kwargs)
        self.assertEqual(report['search'], second['search'])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]))
        self.assertNotEqual(report['benchmarks'], second['benchmarks'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_evaluation_does_not_update_weights(self):
        import torch
        from quantpilot.deep_rl import train, evaluate
        model, _ = train(self.env, episodes=4, batch_size=4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        evaluate(self.env, model)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))

    def test_invalid_parameters(self):
        from quantpilot.deep_rl import TradingMarket, train
        with self.assertRaises(ValueError):
            TradingMarket(self.bars, 1, 8, window=3)
        if importlib.util.find_spec('torch'):
            with self.assertRaises(ValueError):
                train(self.env, gamma=1)


if __name__ == '__main__':
    unittest.main()
