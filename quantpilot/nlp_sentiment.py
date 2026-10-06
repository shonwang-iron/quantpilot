"""Chinese SnowNLP and English VADER adapters with explicit score semantics."""

import math
import re


class NLPSentiment:
    def __init__(self, backend='auto'):
        if backend not in ('auto', 'snownlp', 'vader'):
            raise ValueError('NLP backend 必須為 auto、snownlp 或 vader')
        self.backend = backend
        self._snow = None
        self._vader = None

    def predict_many(self, texts):
        return [self.predict(text) for text in texts]

    def predict(self, text):
        chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        english = bool(re.search(r'[a-zA-Z]', text))
        backend = ('snownlp' if chinese else 'vader') if self.backend == 'auto' else self.backend
        if not text.strip():
            return {'label': 'neutral', 'score': 0.0, 'probabilities': None,
                    'backend': backend, 'status': 'empty_text', 'tokens': []}
        if (backend == 'snownlp' and not chinese) or (backend == 'vader' and chinese):
            raise ValueError('輸入語言不適用指定的 NLP backend；中文請用 snownlp，英文請用 vader')
        try:
            if backend == 'snownlp':
                if self._snow is None:
                    from snownlp import SnowNLP
                    self._snow = SnowNLP
                simplified = self._snow(text).han
                document = self._snow(simplified)
                probability = float(document.sentiments)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('SnowNLP 回傳無效的正面情緒機率')
                score = 2 * probability - 1
                # Neutral is a heuristic band, not a trained third class.
                label = 'positive' if probability > 0.6 else 'negative' if probability < 0.4 else 'neutral'
                return {'label': label, 'score': score, 'positive_probability': probability,
                        'probabilities': None, 'backend': backend, 'tokens': list(document.words),
                        'normalization': 'traditional_to_simplified',
                        'status': 'mixed_language' if english else 'ok'}
            if self._vader is None:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            components = self._vader.polarity_scores(text)
            score = float(components['compound'])
            if not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('VADER 回傳無效的 compound 分數')
            label = 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'
            return {'label': label, 'score': score, 'probabilities': None, 'backend': backend,
                    'components': components, 'tokens': re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text),
                    'normalization': 'none', 'status': 'ok'}
        except ImportError as exc:
            raise RuntimeError('請安裝 NLP 套件：python -m pip install -r requirements-nlp.txt') from exc
