"""Compare sentiment and nonoverlapping, single-asset round-trip strategies."""

import argparse
import csv
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean

from .sentiment import LABELS, NaiveBayes, cluster_news, lexicon_sentiment, read_csv


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('時間必須包含時區，例如 2026-01-05T09:00:00+08:00')
    return result


def load_news(path):
    rows = read_csv(path)
    seen = set()
    result = []
    for row in rows:
        row['_time'] = timestamp(row['available_at'])
        if row.get('label') and row['label'] not in LABELS:
            raise ValueError('新聞 label 必須為三種情緒之一')
    for row in sorted(rows, key=lambda item: item['_time']):
        key = row['text'].strip()
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def load_periods(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('交易期間不可為空')
    for row in rows:
        row['_entry'] = timestamp(row['entry_at'])
        row['_exit'] = timestamp(row['exit_at'])
        row['_open'] = float(row['entry_price'])
        row['_close'] = float(row['exit_price'])
        if row['_entry'] >= row['_exit'] or any(not math.isfinite(row[key]) or row[key] <= 0 for key in ('_open', '_close')):
            raise ValueError('價格必須為有限正數，出場時間必須晚於進場')
    rows.sort(key=lambda row: row['_entry'])
    if any(left['_exit'] > right['_entry'] for left, right in zip(rows, rows[1:])):
        raise ValueError('交易期間不可重疊')
    symbols = {row.get('symbol', '') for row in rows}
    if len(symbols) > 1:
        raise ValueError('每次回測僅支援一個交易標的')
    return rows


def classification(pairs):
    if not pairs:
        return None
    matrix = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    for truth, pred in pairs:
        matrix[truth][pred] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'support': actual,
                            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {'samples': len(pairs), 'accuracy': sum(matrix[label][label] for label in LABELS) / len(pairs),
            'macro_f1': mean(item['f1'] for item in per_class.values()),
            'per_class': per_class, 'confusion_matrix': matrix}


def performance(returns, positions):
    equity = peak = 1.0
    drawdown = 0.0
    for value in returns:
        if value <= -1:
            raise ValueError('模擬資金耗盡；此基準不支援槓桿或破產後繼續交易')
        equity *= 1 + value
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    active = [value for value, position in zip(returns, positions) if position]
    return {'periods': len(returns), 'round_trips': len(active), 'total_return': equity - 1,
            'max_drawdown': drawdown, 'mean_period_return': mean(returns) if returns else None,
            'win_rate': sum(value > 0 for value in active) / len(active) if active else None,
            'exposure_fraction': len(active) / len(returns) if returns else 0.0}


def compare(news, periods, training=None, threshold=0.2, cost_bps=10.0, short=False, deep=None):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold 必須介於 0 與 1')
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError('cost-bps 必須介於 0 與 5000（不含）')
    cutoff = periods[0]['_entry']
    model = None
    if training is not None:
        if any(row['_time'] >= cutoff for row in training):
            raise ValueError('訓練資料 available_at 必須早於第一筆進場時間')
        if {row['text'].strip() for row in training} & {row['text'].strip() for row in news}:
            raise ValueError('訓練與評估新聞不可有相同文字')
        model = NaiveBayes().fit(training)
    methods = ['lexicon', 'cluster'] + (['naive_bayes'] if model else []) + list(deep or {})
    predicted = {name: [] for name in methods}
    pairs = {name: [] for name in methods}
    coverage = {name: 0 for name in methods}
    net = {name: [] for name in methods}
    gross = {name: [] for name in methods}
    positions = {name: [] for name in methods}
    decisions = []
    consumed = set()
    deep_cache = {}
    for name, predictor in (deep or {}).items():
        outputs = predictor.predict_many([row['text'] for row in news])
        if len(outputs) != len(news):
            raise ValueError('深度模型輸出筆數錯誤')
        deep_cache[name] = outputs
    for period in periods:
        eligible = [index for index, row in enumerate(news) if row['_time'] < period['_entry']]
        fresh = [index for index in eligible if index not in consumed]
        consumed.update(fresh)
        # Refit only on news available before this entry; never on the full future batch.
        clusters = cluster_news([news[index]['text'] for index in eligible])
        cluster_map = dict(zip(eligible, clusters))
        market_return = period['_close'] / period['_open'] - 1
        decision = {'entry_at': period['entry_at'], 'exit_at': period['exit_at'],
                    'news_count': len(fresh), 'market_return': market_return, 'methods': {}}
        for name in methods:
            scores = []
            for index in fresh:
                row = news[index]
                if name == 'lexicon':
                    result = lexicon_sentiment(row['text'])
                elif name == 'cluster':
                    item = cluster_map[index]
                    result = {'label': item['interpreted_label'], 'score': item['cluster_lexicon_score'], 'status': 'ok'}
                elif name == 'naive_bayes':
                    result = model.predict(row['text'])
                else:
                    result = deep_cache[name][index]
                scores.append(result['score'])
                predicted[name].append(result['label'])
                if result.get('status') == 'ok':
                    coverage[name] += 1
                if row.get('label'):
                    pairs[name].append((row['label'], result['label']))
            score = mean(scores) if scores else 0.0
            position = 1 if scores and score > threshold else -1 if scores and short and score < -threshold else 0
            before_cost = position * market_return
            # Each interval is an independent trade: enter and fully exit, two one-way costs.
            after_cost = before_cost - 2 * cost_bps / 10000 * abs(position)
            gross[name].append(before_cost)
            net[name].append(after_cost)
            positions[name].append(position)
            decision['methods'][name] = {'score': score, 'position': position,
                                         'gross_return': before_cost, 'net_return': after_cost}
        decisions.append(decision)
    report = {'config': {'threshold': threshold, 'one_way_cost_bps': cost_bps,
                         'strategy': 'long_short' if short else 'long_cash',
                         'execution': 'nonoverlapping_round_trips', 'deduplicated_news': len(news)},
              'methods': {}, 'agreement': {}, 'decisions': decisions,
              'benchmark': {'round_trip_long': performance(
                  [period['_close'] / period['_open'] - 1 - 2 * cost_bps / 10000 for period in periods], [1] * len(periods)),
                  'cash': performance([0.0] * len(periods), [0] * len(periods))}}
    for name in methods:
        report['methods'][name] = {'classification': classification(pairs[name]),
                                   'scored_news': len(predicted[name]), 'untruncated_ok_news': coverage[name],
                                   'gross': performance(gross[name], positions[name]),
                                   'net': performance(net[name], positions[name])}
    for left, right in combinations(methods, 2):
        report['agreement'][left + '__' + right] = (
            mean(a == b for a, b in zip(predicted[left], predicted[right])) if predicted[left] else None)
    return report


def main():
    parser = argparse.ArgumentParser(description='比較新聞情緒方法與單一標的交易基準')
    parser.add_argument('--news', required=True)
    parser.add_argument('--periods', required=True)
    parser.add_argument('--train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--allow-short', action='store_true')
    parser.add_argument('--deep-model', action='append', default=[], help='可重複指定以比較多個模型')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        news = load_news(args.news)
        periods = load_periods(args.periods)
        training = load_news(args.train) if args.train else None
        deep = {}
        if args.deep_model:
            from .deep_sentiment import DeepSentiment
            deep = {'deep:' + name: DeepSentiment(name, args.device) for name in args.deep_model}
        report = compare(news, periods, training, args.threshold, args.cost_bps, args.allow_short, deep)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
