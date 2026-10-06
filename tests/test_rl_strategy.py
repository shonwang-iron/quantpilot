import copy
from pathlib import Path
import unittest

from quantpilot.compare import load_news, load_periods, timestamp
from quantpilot.feature_strategy import build_samples
from quantpilot.rl_strategy import choose, learn, net_return, run, states


class RLTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'examples'
        self.samples = build_samples(load_news(root / 'feature_news_demo.csv'),
                                     load_periods(root / 'feature_periods_demo.csv'))
        self.validation = timestamp('2026-01-23T00:00:00+08:00')
        self.test = timestamp('2026-02-01T00:00:00+08:00')

    def test_reproducible_learning(self):
        first = run(self.samples, self.validation, self.test)
        self.assertEqual(first, run(self.samples, self.validation, self.test))
        self.assertEqual(first[0]['splits']['train'], 6)
        self.assertEqual(first[0]['splits']['test'], 3)

    def test_test_returns_cannot_select_policy(self):
        baseline, policy = run(self.samples, self.validation, self.test)
        changed = copy.deepcopy(self.samples)
        for row in changed[9:]:
            row['return'] = -.4
        report, changed_policy = run(changed, self.validation, self.test)
        self.assertEqual(policy, changed_policy)
        self.assertEqual(baseline['search'], report['search'])
        self.assertNotEqual(baseline['test'], report['test'])

    def test_causal_state_and_unseen_cash(self):
        original = states(self.samples)
        changed = copy.deepcopy(self.samples)
        changed[3]['return'] = -.9
        self.assertEqual(original[:4], states(changed)[:4])
        self.assertEqual(choose({}, 'unknown'), 0)

    def test_round_trip_cost_and_terminal_update(self):
        row = {'return': .02}
        self.assertAlmostEqual(net_return(row, 1, 10), .018)
        self.assertEqual(net_return(row, 0, 10), 0)
        q = learn([row], ['a'], alpha=1, gamma=.9, epsilon=1,
                  episodes=100, seed=42, cost_bps=10)
        self.assertAlmostEqual(q['a'][1], __import__('math').log1p(.018))
        self.assertEqual(q['a'][0], 0)

    def test_cross_boundary_purged(self):
        report, _ = run(self.samples, timestamp('2026-01-21T09:01:00+08:00'), self.test)
        self.assertEqual(report['splits']['purged'], 1)
        self.assertEqual(report['splits']['train'], 5)

    def test_invalid_configuration(self):
        for kwargs in ({'cost_bps': -1}, {'episodes': 0}, {'alphas': (0,)},
                       {'gammas': (1,)}, {'epsilons': (2,)}, {'alphas': ()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run(self.samples, self.validation, self.test, **kwargs)
        with self.assertRaises(ValueError):
            run(self.samples, self.test, self.validation)


if __name__ == '__main__':
    unittest.main()
