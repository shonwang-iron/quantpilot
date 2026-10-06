"""Cached RSS retrieval and grounded company/industry sentiment summaries."""

import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from .compare import timestamp
from .news_data import collect, preprocess, read_records
from .sentiment import lexicon_sentiment

INDUSTRIES = {
    '半導體': ['半導體', '晶圓', '晶片', 'semiconductor', 'chip'],
    '金融': ['金融', '銀行', '保險', '證券', 'bank', 'insurance'],
    '人工智慧': ['人工智慧', '人工智能', 'AI', 'GPU', '大模型'],
    '航運': ['航運', '貨櫃', '海運', 'shipping'],
    '電動車': ['電動車', '電動汽車', 'electric vehicle', 'EV'],
}


def keyword_match(text, word):
    pattern = re.escape(word.casefold())
    if re.search('[a-z0-9]', word.casefold()):
        pattern = r'(?<![a-z0-9])' + pattern + r'(?![a-z0-9])'
    return bool(re.search(pattern, text.casefold()))


class ChatNews:
    def __init__(self, data=None, sources=None, raw_path='data/news_raw.jsonl', predictor=None, limit=5, days=7):
        if not 1 <= limit <= 20 or days < 1:
            raise ValueError('news-limit 需介於 1 與 20，news-days 必須大於 0')
        self.rows = []
        if data:
            self.rows, _ = preprocess(read_records(data))
        self.sources = sources
        self.raw_path = Path(raw_path)
        self.predictor = predictor
        self.limit = limit
        self.days = days

    def refresh(self):
        if not self.sources:
            raise ValueError('未設定 RSS 來源，請提供 --news-sources')
        config = json.loads(Path(self.sources).read_text(encoding='utf-8'))
        if not isinstance(config, list) or not config or any(not isinstance(row, dict) or not row.get('name') or not row.get('url') for row in config):
            raise ValueError('RSS 設定需為含 name,url 的非空陣列')
        rows, statuses = collect(config)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.raw_path.with_name('chat_collection_report.json')
        report.write_text(json.dumps({'sources': statuses, 'collected_records': len(rows)}, ensure_ascii=False, indent=2), encoding='utf-8')
        if not rows:
            raise RuntimeError('本次未取得新聞，未以舊資料冒充更新成功。請查看 ' + str(report))
        with self.raw_path.open('a', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.rows, _ = preprocess([*self.rows, *read_records(self.raw_path)])
        return statuses

    def query(self, terms, subject, refresh=False, now=None, relevance=None):
        statuses = self.refresh() if refresh else []
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.days)
        selected = []
        for row in self.rows:
            acquired = timestamp(row['available_at'])
            dated = timestamp(row['published_at']) if row['published_at'] else acquired
            if (acquired <= now and cutoff <= dated <= now and any(keyword_match(row['text'], term) for term in terms)
                    and (relevance is None or relevance(row['text']))):
                selected.append(row)
        selected.sort(key=lambda row: timestamp(row['published_at'] or row['available_at']), reverse=True)
        selected = selected[:self.limit]
        if not selected:
            return {'answer': f'最近 {self.days} 日的目前新聞資料中，找不到與「{subject}」關鍵字相符的內容；不代表市場沒有相關新聞。',
                    'news': [], 'sentiment_summary': None, 'collection_status': statuses}
        results = (self.predictor.predict_many([row['text'] for row in selected]) if self.predictor else
                   [lexicon_sentiment(row['text']) for row in selected])
        if len(results) != len(selected):
            raise ValueError('情緒結果筆數不一致')
        if any(result['label'] not in ('positive', 'neutral', 'negative') or
               not math.isfinite(result['score']) or not -1 <= result['score'] <= 1 for result in results):
            raise ValueError('情緒模型回傳無效的標籤或分數')
        counts = Counter(result['label'] for result in results)
        summary = {'articles': len(selected), 'positive': counts['positive'], 'neutral': counts['neutral'],
                   'negative': counts['negative'], 'mean_score': mean(result['score'] for result in results),
                   'needs_review': sum(result.get('status') != 'ok' for result in results),
                   'method': type(self.predictor).__name__ if self.predictor else 'lexicon'}
        lines = [f"「{subject}」最近 {self.days} 日的匹配新聞（最新至多 {self.limit} 則）："
                 f"正面 {summary['positive']}／中性 {summary['neutral']}／負面 {summary['negative']}；"
                 f"平均分數 {summary['mean_score']:.3f}，方法 {summary['method']}。",
                 '分布只代表這批標題／摘要的文字情緒，不代表公司報酬或完整產業情緒。']
        if self.predictor and type(self.predictor).__name__ == 'NLPSentiment':
            lines.append('NLP 套件的預訓練模型未針對財經新聞校準，可能誤判財經語境，請檢查原文。')
        if statuses:
            failed = [item['source'] for item in statuses if item['status'] != 'ok']
            lines.append('已更新 RSS，結果包含累積快取。' + ('部分來源失敗：' + '、'.join(failed) if failed else ''))
        else:
            lines.append('使用本機快取；如需重新抓取，問題中加上「抓取最新新聞」。')
        news = []
        for index, (row, result) in enumerate(zip(selected, results), 1):
            news.append({**row, 'sentiment': result})
            review = '（需檢查：' + result['status'] + '）' if result.get('status') != 'ok' else ''
            lines.append(f"{index}. {row['title'] or row['text'][:100]}\n"
                         f"   {result['label']}，分數 {result['score']:.3f}{review}；來源：{row['source']}\n"
                         f"   發布：{row['published_at'] or '未提供'}；取得：{row['available_at']}\n   {row['url'] or '未提供連結'}")
        return {'answer': '\n'.join(lines), 'news': news, 'sentiment_summary': summary, 'collection_status': statuses}
