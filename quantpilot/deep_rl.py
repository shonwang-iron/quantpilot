"""Double DQN with sigmoid price differences and realized-PnL rewards."""

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import random

from .backtest import load_bars
from .compare import timestamp

HOLD, BUY, SELL = 0, 1, 2
ACTIONS = ('hold', 'buy', 'sell')


class TradingMarket:
    """Single unit, long/cash, next-open execution; no leverage or shorting."""

    def __init__(self, bars, start, end, window=5, cash=10000., cost_bps=10., price_scale=1.):
        if window < 1 or start < window + 1 or end > len(bars) or end - start < 2:
            raise ValueError('窗口至少 1；每個分割須有暖機資料與至少兩根可交易價格棒')
        if not math.isfinite(cash) or cash <= 0 or not math.isfinite(price_scale) or price_scale <= 0:
            raise ValueError('cash 與 price-scale 必須為有限正數')
        if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
            raise ValueError('cost-bps 必須在 [0,5000)')
        self.bars, self.start, self.end = bars, start, end
        self.window, self.initial_cash, self.fee = window, cash, cost_bps / 10000
        self.price_scale = price_scale
        self.reset()

    def observation(self):
        import numpy as np
        closes = self.bars.Close.iloc[self.index - self.window - 1:self.index].tolist()
        values = []
        for left, right in zip(closes, closes[1:]):
            difference = (right - left) / self.price_scale
            exponential = math.exp(-abs(difference))
            values.append(1 / (1 + exponential) if difference >= 0 else exponential / (1 + exponential))
        return np.array(values, dtype=np.float32)

    def mask(self):
        import numpy as np
        if self.done:
            return np.array([True, False, False])
        if self.index == self.end - 1:
            return np.array([not self.position, False, bool(self.position)])
        # Affordability uses only observed close; an opening gap can reject a buy.
        affordable = self.cash >= float(self.bars.Close.iloc[self.index - 1]) * (1 + self.fee)
        return np.array([True, not self.position and affordable, bool(self.position)])

    def reset(self):
        self.index, self.cash = self.start, self.initial_cash
        self.position, self.entry_cost, self.done = 0, 0., False
        self.records, self.trades = [], []
        return self.observation(), self.mask()

    def step(self, action):
        if self.done:
            raise ValueError('已終止的環境須先 reset')
        if action not in (HOLD, BUY, SELL) or not self.mask()[action]:
            raise ValueError('無效動作：不允許裸賣、重複買進或末根建立持倉')
        bar = self.bars.iloc[self.index]
        price, reward, filled = float(bar.Open), 0., True
        forced = self.index == self.end - 1 and bool(self.position)
        if action == BUY:
            cost = price * (1 + self.fee)
            if cost <= self.cash:
                self.cash -= cost
                self.position, self.entry_cost = 1, cost
                self.entry_at = self.bars.index[self.index].isoformat()
            else:
                filled = False
        elif action == SELL:
            proceeds = price * (1 - self.fee)
            reward = proceeds - self.entry_cost
            self.cash += proceeds
            self.position = 0
            self.trades.append({'entry_at': self.entry_at,
                                'exit_at': self.bars.index[self.index].isoformat(),
                                'entry_cost': self.entry_cost, 'exit_proceeds': proceeds,
                                'realized_pnl': reward, 'forced_exit': forced})
        equity = self.cash + self.position * float(bar.Close)
        self.records.append({'observation_at': self.bars.close_at.iloc[self.index - 1].isoformat(),
                             'execution_at': self.bars.index[self.index].isoformat(),
                             'action': ACTIONS[action], 'filled': filled, 'price': price,
                             'reward': reward, 'cash': self.cash, 'position': self.position,
                             'equity': equity, 'forced_exit': forced})
        self.index += 1
        self.done = self.index == self.end
        return self.observation(), reward, self.done, self.mask()

    def metrics(self):
        peak, drawdown = self.initial_cash, 0.
        for row in self.records:
            peak = max(peak, row['equity'])
            drawdown = min(drawdown, row['equity'] / peak - 1)
        realized = sum(row['realized_pnl'] for row in self.trades)
        return {'total_return': self.cash / self.initial_cash - 1,
                'realized_pnl': realized, 'final_cash': self.cash,
                'max_drawdown': drawdown, 'trades': len(self.trades),
                'win_rate': (sum(row['realized_pnl'] > 0 for row in self.trades) / len(self.trades)
                             if self.trades else None)}


def network(window, hidden):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(window, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))


def greedy(model, state, mask):
    import torch
    with torch.no_grad():
        values = model(torch.as_tensor(state)).clone()
        values[~torch.as_tensor(mask)] = -torch.inf
        return int(values.argmax().item())


def train(env, *, learning_rate=.001, gamma=.95, hidden=32, episodes=50,
          batch_size=16, buffer_size=10000, target_interval=50, seed=42):
    import numpy as np
    import torch
    if (not math.isfinite(learning_rate) or learning_rate <= 0 or not 0 <= gamma < 1
            or min(hidden, episodes, batch_size, target_interval) < 1 or buffer_size < batch_size):
        raise ValueError('學習率、網路／訓練大小須正數；gamma 在 [0,1)；回放容量須大於批次')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    online, target = network(env.window, hidden), network(env.window, hidden)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    replay, updates, losses = deque(maxlen=buffer_size), 0, []
    initial = {key: value.clone() for key, value in online.state_dict().items()}
    for episode in range(episodes):
        state, mask = env.reset()
        epsilon = max(.05, 1 - episode / max(1, episodes - 1))
        while not env.done:
            action = (rng.choice(np.flatnonzero(mask).tolist()) if rng.random() < epsilon
                      else greedy(online, state, mask))
            next_state, reward, done, next_mask = env.step(action)
            # Raw environment reward stays monetary PnL. Fixed scaling only stabilizes training.
            replay.append((state.copy(), action, reward / env.initial_cash,
                           next_state.copy(), done, next_mask.copy()))
            state, mask = next_state, next_mask
            if len(replay) < batch_size:
                continue
            batch = rng.sample(list(replay), batch_size)
            states_, actions, rewards, next_states, dones, masks = zip(*batch)
            x = torch.tensor(np.stack(states_))
            nx = torch.tensor(np.stack(next_states))
            actions = torch.tensor(actions, dtype=torch.long)
            prediction = online(x).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                # Double DQN: online selects; target evaluates; terminal has no bootstrap.
                next_values = online(nx).masked_fill(~torch.tensor(np.stack(masks)), -torch.inf)
                best = next_values.argmax(1)
                future = target(nx).gather(1, best[:, None]).squeeze(1)
                targets = torch.tensor(rewards, dtype=torch.float32) + gamma * future * (
                    1 - torch.tensor(dones, dtype=torch.float32))
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise ValueError('非有限訓練損失')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.)
            optimizer.step()
            updates += 1
            losses.append(float(loss.item()))
            if updates % target_interval == 0:
                target.load_state_dict(online.state_dict())
    online.eval()
    return online, {'gradient_updates': updates, 'final_loss': losses[-1] if losses else None,
                    'weights_changed': any(not torch.equal(value, initial[key])
                                           for key, value in online.state_dict().items())}


def evaluate(env, model=None, buy_hold=False):
    state, mask = env.reset()
    while not env.done:
        if model is not None:
            action = greedy(model, state, mask)
        elif env.index == env.end - 1 and env.position:
            action = SELL
        elif buy_hold and env.index == env.start and mask[BUY]:
            action = BUY
        else:
            action = HOLD
        state, _, _, mask = env.step(action)
    return env.metrics()


def research(bars, validation_start, test_start, *, window=5, cash=10000., cost_bps=10.,
             price_scale=1., learning_rates=(.001, .0003), gammas=(.9, .99),
             hidden=32, episodes=50, seed=42, batch_size=16):
    if validation_start >= test_start:
        raise ValueError('驗證起點須早於測試起點')
    v = next((i for i, date in enumerate(bars.index) if date >= validation_start), len(bars))
    t = next((i for i, date in enumerate(bars.index) if date >= test_start), len(bars))
    make = lambda start, end: TradingMarket(bars, start, end, window, cash, cost_bps, price_scale)
    training, validation, testing = make(window + 1, v), make(v, t), make(t, len(bars))
    # Ensure historical states and training bars are known before subsequent executions.
    if any(bars.close_at.iloc[i - 1] >= bars.index[i] for i in range(window + 1, len(bars))):
        raise ValueError('前根收盤時間必須嚴格早於下一根成交時間')
    search, best = [], None
    for lr in learning_rates:
        for gamma in gammas:
            params = dict(learning_rate=lr, gamma=gamma, hidden=hidden, episodes=episodes,
                          seed=seed, batch_size=batch_size)
            model, diagnostics = train(training, **params)
            if not diagnostics['weights_changed']:
                raise ValueError('未完成梯度更新，請增加 episodes 或減少 batch-size')
            metrics = evaluate(validation, model)
            objective = metrics['total_return'] - abs(metrics['max_drawdown'])
            search.append({'parameters': params, 'validation': metrics,
                           'objective': objective, 'training': diagnostics})
            if best is None or objective > best[0]:
                best = objective, params, model
    if best is None:
        raise ValueError('參數清單不得為空')
    _, params, model = best
    metrics = evaluate(testing, model)
    report = {'algorithm': 'double_dqn', 'selected_parameters': params, 'search': search,
              'config': {'window': window, 'cash': cash, 'cost_bps': cost_bps,
                         'price_scale': price_scale, 'reward_scale': cash, 'actions': ACTIONS},
              'splits': {'training_bars': v - window - 1, 'validation_bars': t - v,
                         'test_bars': len(bars) - t,
                         'validation_start': validation_start.isoformat(),
                         'test_start': test_start.isoformat(),
                         'train_latest_close': bars.close_at.iloc[v - 1].isoformat()},
              'test': metrics,
              'benchmarks': {'buy_hold_one_unit': evaluate(make(t, len(bars)), buy_hold=True),
                             'cash': evaluate(make(t, len(bars)))},
              'decisions': testing.records, 'trades': testing.trades}
    return report, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', required=True)
    parser.add_argument('--validation-start', required=True)
    parser.add_argument('--test-start', required=True)
    parser.add_argument('--output-dir', required=True)
    for name, default in [('window', 3), ('hidden', 32), ('episodes', 50), ('seed', 42), ('batch-size', 16)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('cash', 10000.), ('cost-bps', 10.), ('price-scale', 1.)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--learning-rates', default='.001,.0003')
    parser.add_argument('--gammas', default='.9,.99')
    args = parser.parse_args()
    try:
        import torch
        report, model = research(load_bars(args.prices), timestamp(args.validation_start),
                                 timestamp(args.test_start), window=args.window, cash=args.cash,
                                 cost_bps=args.cost_bps, price_scale=args.price_scale,
                                 hidden=args.hidden, episodes=args.episodes, seed=args.seed,
                                 batch_size=args.batch_size,
                                 learning_rates=tuple(map(float, args.learning_rates.split(','))),
                                 gammas=tuple(map(float, args.gammas.split(','))))
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report['torch_version'] = torch.__version__
        torch.save({'state_dict': model.state_dict(), 'config': report['config'],
                    'parameters': report['selected_parameters']}, output / 'policy.pt')
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                      allow_nan=False), encoding='utf-8')
        for name in ('decisions', 'trades'):
            rows = report[name]
            fields = list(rows[0]) if rows else ['entry_at', 'exit_at', 'realized_pnl']
            with (output / (name + '.csv')).open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(report['test'], ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
