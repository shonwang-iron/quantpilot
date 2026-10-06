"""Finite-state Q-learning: policy, validation search, frozen-policy backtest."""

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path

from .compare import load_news, load_periods, performance, timestamp
from .feature_strategy import build_samples


def states(samples):
    """Fixed bins need no fitting; only strictly completed returns enter momentum."""
    result = []
    for row in samples:
        history = [x for x in samples if x['period']['_exit'] < row['period']['_entry']]
        momentum = history[-1]['return'] if history else 0
        sentiment = row['features'][0]
        result.append(f"{int(sentiment > .2) - int(sentiment < -.2)}:"
                      f"{int(momentum > .005) - int(momentum < -.005)}:"
                      f"{int(row['news_count'] > 0)}")
    return result


def net_return(row, action, cost_bps):
    # Each long decision opens and closes one full-capital, fractional position.
    return action * (row['return'] - 2 * cost_bps / 10000)


def choose(q, state):
    values = q.get(state, [0.0, 0.0])
    return int(values[1] > values[0])  # Unknown states and ties stay in cash.


def learn(rows, keys, *, alpha, gamma, epsilon, episodes, seed, cost_bps):
    if not 0 < alpha <= 1 or not 0 <= gamma < 1 or not 0 <= epsilon <= 1:
        raise ValueError('alpha 必須在 (0,1]；gamma 在 [0,1)；epsilon 在 [0,1]')
    if episodes < 1 or not rows or len(rows) != len(keys):
        raise ValueError('訓練資料與狀態須非空且等長，episodes 必須正數')
    q = {state: [0.0, 0.0] for state in keys}
    rng = random.Random(seed)
    for _ in range(episodes):
        for i, (row, state) in enumerate(zip(rows, keys)):
            action = rng.randrange(2) if rng.random() < epsilon else choose(q, state)
            value = net_return(row, action, cost_bps)
            if value <= -1:
                raise ValueError('扣成本報酬不得低於或等於 -100%')
            reward = math.log1p(value)
            future = max(q[keys[i + 1]]) if i + 1 < len(keys) else 0.0
            q[state][action] += alpha * (reward + gamma * future - q[state][action])
    return q


def evaluate(rows, keys, q, cost_bps):
    actions = [choose(q, key) for key in keys]
    returns = [net_return(row, action, cost_bps) for row, action in zip(rows, actions)]
    return performance(returns, actions), actions, returns


def run(samples, validation_start, test_start, *, episodes=200, seed=42,
        cost_bps=10.0, alphas=(.1, .3), gammas=(0.0, .9), epsilons=(.1, .3)):
    if validation_start >= test_start:
        raise ValueError('validation-start 必須早於 test-start')
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError('cost-bps 必須在 [0,5000)')
    keys = states(samples)
    train, validation, test = [], [], []
    purged = 0
    for i, row in enumerate(samples):
        entry, exit_at = row['period']['_entry'], row['period']['_exit']
        if entry < validation_start:
            if exit_at < validation_start:
                train.append(i)
            else:
                purged += 1
        elif entry < test_start:
            if exit_at < test_start:
                validation.append(i)
            else:
                purged += 1
        else:
            test.append(i)
    if min(map(len, (train, validation, test))) < 2:
        raise ValueError('訓練、驗證及測試各須至少 2 個完整期間')
    select = lambda indices: ([samples[i] for i in indices], [keys[i] for i in indices])
    train_rows, train_keys = select(train)
    validation_rows, validation_keys = select(validation)
    candidates = []
    best = None
    for alpha, gamma, epsilon in itertools.product(alphas, gammas, epsilons):
        params = dict(alpha=alpha, gamma=gamma, epsilon=epsilon,
                      episodes=episodes, seed=seed, cost_bps=cost_bps)
        q = learn(train_rows, train_keys, **params)
        metrics, _, _ = evaluate(validation_rows, validation_keys, q, cost_bps)
        objective = metrics['total_return'] - abs(metrics['max_drawdown'])
        candidates.append({'parameters': params, 'validation': metrics, 'objective': objective})
        if best is None or objective > best[0]:
            best = (objective, params, q)
    if best is None:
        raise ValueError('參數搜尋清單不得為空')
    _, params, q = best
    test_rows, test_keys = select(test)
    metrics, actions, returns = evaluate(test_rows, test_keys, q, cost_bps)
    decisions = []
    equity = 1.0
    for row, key, action, value in zip(test_rows, test_keys, actions, returns):
        equity *= 1 + value
        decisions.append({'entry_at': row['period']['entry_at'],
                          'exit_at': row['period']['exit_at'], 'state': key,
                          'action': action, 'gross_return': action * row['return'],
                          'net_return': value, 'equity': equity})
    long_returns = [net_return(row, 1, cost_bps) for row in test_rows]
    report = {'algorithm': 'tabular_q_learning', 'selected_parameters': params,
              'splits': {'train': len(train), 'validation': len(validation),
                         'test': len(test), 'purged': purged,
                         'validation_start': validation_start.isoformat(),
                         'test_start': test_start.isoformat()},
              'search': candidates, 'test': metrics,
              'benchmarks': {'long_every_period': performance(long_returns, [1] * len(test)),
                             'cash': performance([0.0] * len(test), [0] * len(test))},
              'decisions': decisions}
    policy = {'algorithm': 'tabular_q_learning', 'parameters': params, 'q_table': q,
              'state': {'sentiment_threshold': .2, 'momentum_threshold': .005,
                        'components': ['sentiment_bin', 'last_completed_return_bin', 'has_news']},
              'actions': {'0': 'cash', '1': 'long_round_trip'},
              'reward': 'log(1 + action * (period_return - 2 * cost_bps / 10000))',
              'train_latest_exit': train_rows[-1]['period']['exit_at']}
    return report, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--news', required=True)
    parser.add_argument('--periods', required=True)
    parser.add_argument('--validation-start', required=True)
    parser.add_argument('--test-start', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cost-bps', type=float, default=10)
    for name, default in [('alphas', '.1,.3'), ('gammas', '0,.9'), ('epsilons', '.1,.3')]:
        parser.add_argument('--' + name, default=default)
    args = parser.parse_args()
    try:
        grid = {name: tuple(float(x) for x in getattr(args, name).split(','))
                for name in ('alphas', 'gammas', 'epsilons')}
        report, policy = run(build_samples(load_news(args.news), load_periods(args.periods)),
                             timestamp(args.validation_start), timestamp(args.test_start),
                             episodes=args.episodes, seed=args.seed, cost_bps=args.cost_bps, **grid)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in [('report', report), ('policy', policy)]:
            (output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                            allow_nan=False), encoding='utf-8')
        with (output / 'decisions.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(report['decisions'][0]))
            writer.writeheader()
            writer.writerows(report['decisions'])
        print(json.dumps(report['test'], ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
