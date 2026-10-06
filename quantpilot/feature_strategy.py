"""Causal expanding-window return prediction with feature ablation."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from .compare import load_news, load_periods, performance, timestamp
from .sentiment import lexicon_sentiment, tokenize

FEATURE_NAMES = ['sentiment_mean', 'sentiment_dispersion', 'positive_share',
                 'negative_share', 'neutral_share', 'log_news_count', 'valid_share']


def build_samples(news, periods, predictor=None):
    """Consume each article only at the first strictly later entry."""
    predictions = (predictor.predict_many([row['text'] for row in news]) if predictor
                   else [lexicon_sentiment(row['text']) for row in news])
    if len(predictions) != len(news):
        raise ValueError('情緒模型回傳筆數與新聞不一致')
    consumed = set()
    samples = []
    for period in periods:
        fresh = [i for i, row in enumerate(news) if i not in consumed and row['_time'] < period['_entry']]
        consumed.update(fresh)
        results = [predictions[i] for i in fresh]
        scores = [result['score'] for result in results]
        if any(not math.isfinite(score) or not -1 <= score <= 1 for score in scores):
            raise ValueError('情緒分數必須為 -1 到 1 的有限數值')
        average = mean(scores) if scores else 0.0
        features = [average, math.sqrt(mean((score - average)**2 for score in scores)) if scores else 0.0]
        features += [sum(result['label'] == label for result in results) / len(results) if results else 0.0
                     for label in ('positive', 'negative', 'neutral')]
        features += [math.log1p(len(fresh)), sum(result.get('status') == 'ok' for result in results) / len(results) if results else 0.0]
        samples.append({'period': period, 'text': '\n'.join(news[i]['text'] for i in fresh),
                        'features': features, 'news_count': len(fresh),
                        'return': period['_close'] / period['_open'] - 1})
    return samples


def fit_predict(training, current, alpha=1.0, max_features=2000):
    """All vocabulary, IDF, scaling, and regressors are fitted on training only."""
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 max_features=max_features, sublinear_tf=True)
    if any(tokenize(row['text']) for row in training):
        words_train = vectorizer.fit_transform([row['text'] for row in training])
        words_test = vectorizer.transform([current['text']])
        vocabulary = vectorizer.get_feature_names_out().tolist()
    else:
        words_train = csr_matrix((len(training), 1))
        words_test = csr_matrix((1, 1))
        vocabulary = []
    scaler = StandardScaler()
    sentiment_train = csr_matrix(scaler.fit_transform([row['features'] for row in training]))
    sentiment_test = csr_matrix(scaler.transform([current['features']]))
    designs = {'sentiment_only': (sentiment_train, sentiment_test),
               'words_only': (words_train, words_test),
               'combined': (hstack([sentiment_train, words_train], format='csr'),
                            hstack([sentiment_test, words_test], format='csr'))}
    predictions = {}
    for name, (train, test) in designs.items():
        regressor = Ridge(alpha=alpha, solver='lsqr', tol=1e-8)
        regressor.fit(train, np.array([row['return'] for row in training]))
        predictions[name] = float(regressor.predict(test)[0])
        if not math.isfinite(predictions[name]):
            raise ValueError('模型產生非有限的預測報酬，停止回測')
    return predictions, {'vocabulary_size': len(vocabulary), 'train_periods': len(training)}


def backtest(samples, test_start, min_train=30, alpha=1.0, max_features=2000,
             cost_bps=10.0, edge=0.001):
    if min_train < 2 or max_features < 1:
        raise ValueError('min-train 必須至少 2，max-features 必須大於 0')
    if any(not math.isfinite(value) or value < 0 for value in (cost_bps, edge, alpha)) or alpha == 0 or cost_bps >= 5000:
        raise ValueError('alpha 必須大於 0；edge 非負；cost-bps 介於 0 與 5000（不含）')
    methods = ('sentiment_only', 'words_only', 'combined', 'historical_mean', 'long_every_period', 'cash')
    net = {name: [] for name in methods}
    gross = {name: [] for name in methods}
    positions = {name: [] for name in methods}
    errors = {name: [] for name in methods[:4]}
    decisions = []
    cost = 2 * cost_bps / 10000
    skipped = 0
    for current in samples:
        period = current['period']
        if period['_entry'] < test_start:
            continue
        # Exit return must be fully known strictly before this entry.
        training = [row for row in samples if row['period']['_exit'] < period['_entry']]
        if len(training) < min_train:
            skipped += 1
            continue
        predictions, metadata = fit_predict(training, current, alpha, max_features)
        predictions['historical_mean'] = mean(row['return'] for row in training)
        predictions.update({'long_every_period': None, 'cash': None})
        decision = {'entry_at': period['entry_at'], 'exit_at': period['exit_at'],
                    'train_latest_exit': max(row['period']['_exit'] for row in training).isoformat(),
                    'news_count': current['news_count'], 'features': dict(zip(FEATURE_NAMES, current['features'])),
                    'actual_return': current['return'], **metadata, 'methods': {}}
        for name in methods:
            prediction = predictions[name]
            position = (1 if name == 'long_every_period' else 0 if name == 'cash' else
                        int(current['news_count'] > 0 and prediction > cost + edge))
            before_cost = position * current['return']
            after_cost = before_cost - cost * position
            gross[name].append(before_cost)
            net[name].append(after_cost)
            positions[name].append(position)
            if prediction is not None:
                errors[name].append(prediction - current['return'])
            decision['methods'][name] = {'predicted_return': prediction, 'position': position,
                                        'gross_return': before_cost, 'net_return': after_cost}
        decisions.append(decision)
    if not decisions:
        raise ValueError('沒有足夠已完成的歷史期間可訓練；請增加資料或降低 min-train（正式研究需足量資料）')
    report = {'config': {'test_start': test_start.isoformat(), 'min_train': min_train, 'alpha': alpha,
                        'max_features': max_features, 'one_way_cost_bps': cost_bps, 'minimum_net_edge': edge,
                        'position': 'long_cash', 'training': 'expanding_window', 'skipped_test_periods': skipped},
              'methods': {}, 'decisions': decisions}
    for name in methods:
        report['methods'][name] = {'gross': performance(gross[name], positions[name]),
                                   'net': performance(net[name], positions[name])}
        if name in errors:
            report['methods'][name]['prediction_mae'] = mean(abs(error) for error in errors[name])
            report['methods'][name]['prediction_rmse'] = math.sqrt(mean(error**2 for error in errors[name]))
    return report


def main():
    parser = argparse.ArgumentParser(description='情緒與詞彙向量交易特徵：逐期訓練與消融比較')
    parser.add_argument('--news', required=True)
    parser.add_argument('--periods', required=True)
    parser.add_argument('--test-start', required=True, help='含時區的測試起始時間')
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-train', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--max-features', type=int, default=2000)
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--edge', type=float, default=0.001)
    parser.add_argument('--deep-model', help='省略時使用詞典情緒；指定時使用 Transformer 情緒')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        predictor = None
        if args.deep_model:
            from .deep_sentiment import DeepSentiment
            predictor = DeepSentiment(args.deep_model, args.device)
        samples = build_samples(load_news(args.news), load_periods(args.periods), predictor)
        report = backtest(samples, timestamp(args.test_start), args.min_train, args.alpha,
                          args.max_features, args.cost_bps, args.edge)
        report['config']['sentiment_model'] = args.deep_model or 'lexicon'
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    except ImportError as exc:
        parser.error('缺少套件，請安裝 requirements-strategy.txt；深度情緒另需 requirements-deep.txt：' + str(exc))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
