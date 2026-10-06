"""OHLC event-driven execution using pandas and backtesting.py."""

import argparse
import csv
import json
import math
from importlib.metadata import version
from pathlib import Path

from .compare import timestamp


def load_bars(path):
    import pandas as pd
    bars = pd.read_csv(path)
    required = {'open_at', 'close_at', 'Open', 'High', 'Low', 'Close'}
    if not required.issubset(bars.columns):
        raise ValueError('價格 CSV 必須包含 open_at,close_at,Open,High,Low,Close')
    if len(bars) < 4:
        raise ValueError('至少需要四根價格棒（含第一根暖機價格棒）')
    for column in ('open_at', 'close_at'):
        bars[column] = pd.to_datetime([timestamp(value) for value in bars[column]], utc=True)
    bars = bars.sort_values('open_at').set_index('open_at')
    if bars.index.has_duplicates:
        raise ValueError('價格棒時間不可重複')
    if 'symbol' in bars and bars['symbol'].nunique(dropna=False) != 1:
        raise ValueError('每次回測僅支援單一標的')
    for column in ('Open', 'High', 'Low', 'Close'):
        bars[column] = pd.to_numeric(bars[column], errors='raise')
        if any(not math.isfinite(value) or value <= 0 for value in bars[column]):
            raise ValueError('OHLC 必須為有限正數')
    if any(bars.High < bars[['Open', 'Close', 'Low']].max(axis=1)) or any(bars.Low > bars[['Open', 'Close', 'High']].min(axis=1)):
        raise ValueError('OHLC 高低價不一致')
    for index, (opened, row) in enumerate(bars.iterrows()):
        if opened >= row.close_at or (index + 1 < len(bars) and row.close_at > bars.index[index + 1]):
            raise ValueError('價格棒起訖時間無效或互相重疊')
    if 'Volume' in bars:
        bars['Volume'] = pd.to_numeric(bars.Volume, errors='raise')
        if any(not math.isfinite(value) or value < 0 for value in bars.Volume):
            raise ValueError('Volume 必須為有限非負數')
    return bars


def load_signals(path, method=None):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == '.json':
        report = json.loads(path.read_text(encoding='utf-8-sig'))
        if not method:
            raise ValueError('策略比較 JSON 必須指定 --method，例如 combined')
        rows = []
        for decision in report['decisions']:
            result = decision['methods'][method]
            score = result.get('predicted_return', result.get('score'))
            if score is None:
                raise ValueError('所選策略沒有可用的預測或情緒分數')
            # Conservative availability: never claim a prediction existed before its decision.
            rows.append({'available_at': decision['entry_at'], 'score': score})
    else:
        with path.open(encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not method:
            raise ValueError('情緒 JSONL 必須指定 --method，例如 nlp_sentiment')
        converted = []
        for row in rows:
            result = row.get(method)
            if result is None:
                raise ValueError('所選情緒方法未執行：' + method)
            converted.append({'available_at': row['available_at'], 'score': result['score']})
        rows = converted
    for row in rows:
        row['_time'] = timestamp(row['available_at'])
        row['score'] = float(row['score'])
        if not math.isfinite(row['score']):
            raise ValueError('訊號分數必須為有限數值')
    return sorted(rows, key=lambda row: row['_time'])


def align_signals(bars, signals):
    from statistics import mean
    bars = bars.copy()
    values = []
    counts = []
    pointer = 0
    stale = 0
    while pointer < len(signals) and signals[pointer]['_time'] < bars.index[0]:
        pointer += 1
        stale += 1
    for close in bars.close_at:
        available = []
        while pointer < len(signals) and signals[pointer]['_time'] < close:
            available.append(signals[pointer]['score'])
            pointer += 1
        values.append(mean(available) if available else float('nan'))
        counts.append(len(available))
    bars['Signal'] = values
    bars['NewsCount'] = counts
    return bars, {'signals': len(signals), 'before_price_range': stale,
                  'warmup_signals': counts[0], 'after_last_close': len(signals) - pointer}


def run_backtest(bars, cash=1000000, cost_bps=10, threshold=0.2, hold_bars=1, allocation=0.9):
    from backtesting import Backtest, Strategy
    if any(not math.isfinite(value) for value in (cash, cost_bps, threshold, allocation)) or cash <= 0 or not 0 <= cost_bps < 1000 or not 0 < allocation < 1 or not isinstance(hold_bars, int) or hold_bars < 1:
        raise ValueError('cash 必須正值；cost-bps 介於 0 與 1000（不含）；allocation 介於 0 與 1（不含）；hold-bars 至少 1')
    if len(bars) < hold_bars + 3:
        raise ValueError('價格棒不足以暖機、進場及完成持有期間')

    class NewsSignalStrategy(Strategy):
        def init(self):
            self.total_bars = len(self.data)

        def next(self):
            current = len(self.data) - 1
            if self.position:
                if current - self.trades[0].entry_bar >= hold_bars - 1:
                    self.position.close()
                return
            # Only allow entries that can fully exit at a later real bar open.
            if current > self.total_bars - hold_bars - 2:
                return
            score = self.data.Signal[-1]
            if math.isfinite(score) and score > threshold:
                self.buy(size=allocation)

    engine = Backtest(bars, NewsSignalStrategy, cash=cash, commission=cost_bps / 10000,
                      trade_on_close=False, exclusive_orders=True, finalize_trades=False)
    stats = engine.run()
    selected = ['Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]', '# Trades',
                'Win Rate [%]', 'Profit Factor', 'Exposure Time [%]', 'Equity Final [$]']
    metrics = {}
    for name in selected:
        value = float(stats[name])
        metrics[name] = value if math.isfinite(value) else None
    report = {'engine': 'backtesting.py', 'versions': {'backtesting': version('backtesting'), 'pandas': version('pandas')}, 'config': {'cash': cash, 'one_way_cost_bps': cost_bps,
        'threshold': threshold, 'hold_bars': hold_bars, 'allocation': allocation,
        'execution': 'next_bar_open', 'position': 'long_cash', 'warmup_bars': 1}, 'metrics': metrics}
    return report, stats['_trades'], stats['_equity_curve']


def main():
    parser = argparse.ArgumentParser(description='backtesting.py 單一標的新聞訊號回測框架')
    parser.add_argument('--prices', required=True)
    parser.add_argument('--signals', required=True)
    parser.add_argument('--method', help='情緒 JSONL 的方法欄位')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cash', type=float, default=1000000)
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--hold-bars', type=int, default=1)
    parser.add_argument('--allocation', type=float, default=0.9)
    args = parser.parse_args()
    try:
        bars, quality = align_signals(load_bars(args.prices), load_signals(args.signals, args.method))
        report, trades, equity = run_backtest(bars, args.cash, args.cost_bps, args.threshold, args.hold_bars, args.allocation)
        report['signal_quality'] = quality
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        trades.to_csv(output / 'trades.csv', index=False, encoding='utf-8-sig')
        equity.to_csv(output / 'equity.csv', encoding='utf-8-sig')
        bars[['close_at', 'Signal', 'NewsCount']].to_csv(output / 'aligned_signals.csv', encoding='utf-8-sig')
    except ImportError as exc:
        parser.error('請安裝 requirements-backtest.txt：' + str(exc))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
