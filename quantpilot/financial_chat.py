"""Grounded financial-ratio conversation prototype with optional SnowNLP."""

import argparse
import csv
import json
import math
import calendar
import re
import unicodedata
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from .compare import timestamp

TWSE_URL = 'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL'
CATALOG = {
    'pe': ('本益比', ('本益比', '市盈率', 'pe', 'p/e'), '倍', '股價相對每股盈餘的倍數；不可單靠此值判定買賣。'),
    'pb': ('股價淨值比', ('股價淨值比', '淨值比', '市净率', 'pb', 'p/b'), '倍', '股價相對每股帳面淨值的倍數。'),
    'dividend_yield': ('殖利率', ('殖利率', '股息率', 'dividend yield'), '%', '股息相對股價的比例；此資料不是未來股息保證。'),
    'roe': ('股東權益報酬率', ('股東權益報酬率', '股东权益报酬率', 'roe'), '%', '衡量獲利相對股東權益的比例，需確認期間與計算口徑。'),
    'gross_margin': ('毛利率', ('毛利率', 'gross margin'), '%', '毛利占營收的比例。'),
    'operating_margin': ('營業利益率', ('營業利益率', '營益率', 'operating margin'), '%', '營業利益占營收的比例。'),
    'net_margin': ('淨利率', ('淨利率', '净利率', 'net margin'), '%', '淨利占營收的比例，需確認歸屬口徑。'),
    'eps': ('每股盈餘', ('每股盈餘', '每股收益', 'eps'), '元/股', '每股盈餘須區分基本／稀釋與期間，不同口徑不可直接比較。'),
    'current_ratio': ('流動比率', ('流動比率', '流动比率', 'current ratio'), '倍', '流動資產相對流動負債的比例。'),
    'debt_ratio': ('負債比率', ('負債比率', '負債比', 'debt ratio'), '%', '此介面採總負債占總資產比例，與負債權益比不同。'),
}


def normalize(text):
    return unicodedata.normalize('NFKC', text).casefold().strip()


def matches_alias(text, alias):
    alias = normalize(alias)
    if re.search('[a-z0-9]', alias):
        return bool(re.search(r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])', text))
    return alias in text


def period_end(value):
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return date.fromisoformat(value)
    match = re.fullmatch(r'(\d{4})-?Q([1-4])', value)
    if match:
        year, quarter = map(int, match.groups())
        month = quarter * 3
        return date(year, month, calendar.monthrange(year, month)[1])
    raise ValueError('period 必須為 YYYY-MM-DD 或 YYYY-Q1 至 YYYY-Q4')


def validate_records(rows):
    result = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        for field in ('symbol', 'company', 'metric', 'period', 'available_at', 'source_url', 'unit'):
            if not row.get(field):
                raise ValueError('財務資料缺少欄位：' + field)
        if row['metric'] not in CATALOG:
            raise ValueError('不支援的財務指標：' + row['metric'])
        if row['unit'] != CATALOG[row['metric']][2]:
            raise ValueError('指標單位不符：' + row['metric'])
        timestamp(row['available_at'])
        period_end(row['period'])
        if not row['source_url'].startswith('https://'):
            raise ValueError('財務資料必須提供 HTTPS 來源連結')
        raw_value = row.get('value')
        row['value'] = None if raw_value in (None, '', '-', '--', 'N/A') else float(raw_value)
        if row['value'] is not None and not math.isfinite(row['value']):
            raise ValueError('財務數值必須為有限數值')
        key = (row['symbol'], row['metric'], row['period'])
        if key in seen:
            raise ValueError('同一公司／指標／期間資料重複，請先確認口徑')
        seen.add(key)
        row.setdefault('aliases', '')
        row.setdefault('is_demo', False)
        row['is_demo'] = str(row['is_demo']).lower() in ('true', '1', 'yes')
        row.setdefault('basis', '')
        result.append(row)
    if not result:
        raise ValueError('財務資料不可為空')
    return result


def convert_twse(rows, obtained):
    records = []
    for row in rows:
        raw_date = str(row['Date'])
        if not re.fullmatch(r'\d{7}', raw_date):
            raise ValueError('證交所資料日期格式改變，停止解析')
        period = date(int(raw_date[:3]) + 1911, int(raw_date[3:5]), int(raw_date[5:])).isoformat()
        for field, metric in (('PEratio', 'pe'), ('PBratio', 'pb'), ('DividendYield', 'dividend_yield')):
            records.append({'symbol': str(row['Code']), 'company': row['Name'], 'metric': metric,
                'value': row[field], 'unit': CATALOG[metric][2], 'period': period,
                'available_at': obtained, 'source_url': TWSE_URL, 'basis': 'TWSE 官方日資料口徑'})
    return validate_records(records)


def fetch_twse():
    request = urllib.request.Request(TWSE_URL, headers={'User-Agent': 'Quantpilot financial research prototype'})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError('證交所回應超過大小限制')
    return convert_twse(json.loads(payload), datetime.now(timezone.utc).isoformat())


def read_data(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return validate_records(list(csv.DictReader(stream)))
    return validate_records(json.loads(path.read_text(encoding='utf-8-sig')))


class FinancialChat:
    def __init__(self, records, nlp='simple', news_service=None):
        self.records = validate_records(records)
        self.company_context = []
        self.nlp = nlp
        self.news_service = news_service
        self.news_context = None
        if nlp == 'snownlp':
            from snownlp import SnowNLP
            self.segment = lambda text: list(SnowNLP(text).words)
        else:
            self.segment = lambda text: re.findall(r'[a-z]+|[\u4e00-\u9fff]+|\d+', normalize(text))
        self.companies = {}
        for row in self.records:
            company = self.companies.setdefault(row['symbol'], {'name': row['company'], 'aliases': set()})
            company['aliases'].update([row['symbol'], row['company']])
            company['aliases'].update(alias for alias in row.get('aliases', '').split('|') if alias)

    def resolve_companies(self, text):
        candidates = []
        for symbol, company in self.companies.items():
            for alias in company['aliases']:
                alias = normalize(alias)
                pattern = re.escape(alias)
                if re.search('[a-z0-9]', alias):
                    pattern = r'(?<![a-z0-9])' + pattern + r'(?![a-z0-9])'
                for match in re.finditer(pattern, text):
                    candidates.append((match.start(), match.end(), symbol))
        selected = []
        by_span = {}
        for start, end, symbol in candidates:
            by_span.setdefault((start, end), set()).add(symbol)
        for start, end, symbol in sorted(candidates, key=lambda item: item[1]-item[0], reverse=True):
            if not any(start < right and end > left for left, right, _ in selected):
                if len(by_span[(start, end)]) > 1:
                    raise ValueError('公司別名對應多家公司，請改用明確股票代號。')
                selected.append((start, end, symbol))
        return list(dict.fromkeys(symbol for _, _, symbol in sorted(selected)))

    def reply(self, question):
        text = normalize(question)
        if not text:
            return {'answer': '請輸入公司名稱／代號與財務比率，例如「2330 本益比」。', 'facts': []}
        if text in ('重設', 'reset', '清除對話'):
            self.company_context = []
            self.news_context = None
            return {'answer': '已清除公司脈絡。', 'facts': []}
        metrics = [key for key, (_, aliases, _, _) in CATALOG.items() if any(matches_alias(text, alias) for alias in aliases)]
        try:
            symbols = self.resolve_companies(text)
        except ValueError as exc:
            return {'answer': str(exc), 'facts': []}
        explicit_codes = re.findall(r'(?<![a-z0-9])\d{4,6}(?![a-z0-9])', text)
        history = bool(re.search(r'\d{4}[年/-]|第.{1,2}季|去年|前年|上季|last year|quarter', text))
        if history:
            return {'answer': '目前原型尚未支援歷史／相對期間查詢，請先明確指定資料期間；不以最新值冒充歷史值。', 'facts': []}
        unknown = [symbol for symbol in explicit_codes if symbol not in self.companies]
        if unknown:
            return {'answer': '目前資料找不到代號 ' + '、'.join(unknown) + '；請確認市場與資料來源。', 'facts': []}
        news_intent = any(word in text for word in ('新聞', '情緒', 'news', 'sentiment'))
        if news_intent:
            if metrics:
                return {'answer': '請分開查詢財務指標與新聞情緒，例如先問「台積電本益比」，再問「那新聞情緒呢？」。', 'facts': []}
            return self.reply_news(text, symbols)
        definition = any(word in text for word in ('什麼', '意思', '定義', '解釋', 'what is', 'definition'))
        parsed = {'tokens': self.segment(question), 'metrics': metrics, 'companies': symbols, 'nlp': self.nlp}
        if definition and metrics:
            return {'answer': '\n'.join(CATALOG[metric][0] + '：' + CATALOG[metric][3] for metric in metrics), 'facts': [], 'parsed': parsed}
        if symbols:
            self.company_context = symbols
        else:
            # Only an explicit follow-up may reuse context; never substitute an unknown company name.
            remaining = text
            for metric in metrics:
                for alias in sorted(CATALOG[metric][1], key=len, reverse=True):
                    remaining = remaining.replace(normalize(alias), '')
            remaining = re.sub(r'這家公司|該公司|what about|是多少|多少|如何|please|its|那|它|呢|的', '', remaining)
            followup = bool(metrics and not re.sub(r'[\s?？!！,，。:：]+', '', remaining))
            symbols = self.company_context if followup else []
        if not symbols:
            return {'answer': '請指定資料中可辨識的公司名稱或代號；可先問「2330 本益比」，再接著問「那殖利率呢？」。', 'facts': [], 'parsed': parsed}
        if not metrics:
            if any(word in text for word in ('買', '賣', 'buy', 'sell')):
                answer = '財務比率不能單獨決定買賣；可查詢本益比、股價淨值比、殖利率或匯入其他財務指標。'
            else:
                answer = '請指定財務指標：' + '、'.join(value[0] for value in CATALOG.values()) + '。'
            return {'answer': answer, 'facts': [], 'parsed': parsed}
        facts, lines = [], []
        for symbol in symbols:
            for metric in metrics:
                candidates = [row for row in self.records if row['symbol'] == symbol and row['metric'] == metric]
                if not candidates:
                    lines.append(self.companies[symbol]['name'] + '（' + symbol + '）缺少' + CATALOG[metric][0] + '資料。')
                    continue
                row = max(candidates, key=lambda item: (period_end(item['period']), timestamp(item['available_at'])))
                facts.append(row)
                value = '來源未提供有效值' if row['value'] is None else f"{row['value']:g}{row['unit']}"
                demo = '【合成示範資料】' if row['is_demo'] else ''
                lines.append(f"{demo}{row['company']}（{symbol}）{CATALOG[metric][0]}：{value}；期間：{row['period']}。\n"
                             f"口徑：{row['basis'] or '匯入資料，請確認計算口徑'}；取得：{row['available_at']}\n來源：{row['source_url']}")
        return {'answer': '\n\n'.join(lines), 'facts': facts, 'parsed': parsed}

    def reply_news(self, text, symbols):
        from .chat_news import INDUSTRIES
        if self.news_service is None:
            return {'answer': '新聞功能尚未設定，請提供 --news-data 或 --news-sources。', 'facts': []}
        industry_text = text
        for symbol in symbols:
            for alias in sorted(self.companies[symbol]['aliases'], key=len, reverse=True):
                industry_text = industry_text.replace(normalize(alias), '')
        industries = [name for name in INDUSTRIES if name in industry_text or
                      (name == '人工智慧' and matches_alias(industry_text, 'AI'))]
        if symbols and industries:
            return {'answer': '請一次指定公司或產業，例如「台積電新聞」或「半導體產業新聞」。', 'facts': []}
        target_symbol = None
        if symbols:
            if len(symbols) > 1:
                return {'answer': '請先查詢一家公司的新聞，避免把多家公司情緒混在一起。', 'facts': []}
            company = self.companies[symbols[0]]
            terms = sorted(company['aliases'])
            subject = company['name'] + '（' + symbols[0] + '）'
            self.company_context = symbols
            target_symbol = symbols[0]
            self.news_context = (terms, subject, target_symbol)
        elif industries:
            if len(industries) != 1:
                return {'answer': '請一次指定一個產業。', 'facts': []}
            subject = industries[0]
            terms = INDUSTRIES[subject]
            self.company_context = []
            self.news_context = (terms, subject, None)
        else:
            remaining = re.sub(r'這家公司|該公司|抓取|抓|最新|相關|新聞|情緒|分析|更新|幫我|看看|如何|那|它|的|呢|[\s?？!！,，。]+', '', text)
            if remaining:
                return {'answer': '請指定可辨識的公司名稱／代號，或半導體、金融、人工智慧、航運、電動車產業。', 'facts': []}
            if self.company_context:
                if len(self.company_context) != 1:
                    return {'answer': '目前有多家公司脈絡，請指定其中一家查詢新聞。', 'facts': []}
                company = self.companies[self.company_context[0]]
                terms, subject = sorted(company['aliases']), company['name']
                target_symbol = self.company_context[0]
            elif self.news_context:
                terms, subject, target_symbol = self.news_context
            else:
                return {'answer': '請先指定公司或產業，再追問新聞與情緒。', 'facts': []}
        try:
            def company_relevance(article):
                from .chat_news import keyword_match
                names = [alias for alias in self.companies[target_symbol]['aliases'] if alias != target_symbol]
                named = any(keyword_match(article, alias) for alias in names)
                code = re.escape(target_symbol)
                explicit_code = bool(re.search(r'(?:股票代號|代號|ticker)\s*[:：]?\s*' + code +
                    r'(?!\d)|[（(]\s*' + code + r'\s*[）)]|' + code + r'\.TW\b', article, re.I))
                return (named or explicit_code) and target_symbol in self.resolve_companies(normalize(article))
            relevance = company_relevance if target_symbol else None
            result = self.news_service.query(terms, subject, refresh=any(word in text for word in ('抓', '更新', '最新')), relevance=relevance)
            return {**result, 'facts': []}
        except (ValueError, OSError, RuntimeError) as exc:
            return {'answer': '新聞查詢失敗：' + str(exc), 'facts': [], 'news': []}


def main():
    parser = argparse.ArgumentParser(description='NLP 財務比率對話聊天機器人雛形')
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--data', help='標準財務 CSV／JSON')
    sources.add_argument('--twse', action='store_true', help='使用證交所上市股票官方日資料')
    parser.add_argument('--cache', default='data/financial_ratios.json')
    parser.add_argument('--refresh', action='store_true', help='重新取得證交所資料；失敗時不偷偷回退快取')
    parser.add_argument('--nlp', choices=('simple', 'snownlp'), default='snownlp')
    parser.add_argument('--question', help='單次問題；省略則互動對話')
    parser.add_argument('--news-data', help='已清理的新聞 CSV／原始 JSONL')
    parser.add_argument('--news-sources', help='RSS 來源 JSON；允許對話中抓取最新新聞')
    parser.add_argument('--news-raw', default='data/news_raw.jsonl')
    parser.add_argument('--news-limit', type=int, default=5)
    parser.add_argument('--news-days', type=int, default=7)
    sentiment = parser.add_mutually_exclusive_group()
    sentiment.add_argument('--news-nlp-backend', choices=('auto', 'snownlp', 'vader'))
    sentiment.add_argument('--news-deep-model')
    args = parser.parse_args()
    try:
        if args.twse:
            cache = Path(args.cache)
            records = fetch_twse() if args.refresh or not cache.exists() else read_data(cache)
            if args.refresh or not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            records = read_data(args.data)
        service = None
        if args.news_data or args.news_sources:
            from .chat_news import ChatNews
            predictor = None
            if args.news_nlp_backend:
                from .nlp_sentiment import NLPSentiment
                predictor = NLPSentiment(args.news_nlp_backend)
            elif args.news_deep_model:
                from .deep_sentiment import DeepSentiment
                predictor = DeepSentiment(args.news_deep_model)
            service = ChatNews(args.news_data, args.news_sources, args.news_raw, predictor, args.news_limit, args.news_days)
        bot = FinancialChat(records, args.nlp, service)
        if args.question:
            print(bot.reply(args.question)['answer'])
        else:
            print('財務／新聞聊天雛形；輸入公司比率、公司或產業新聞。exit 離開，重設 清除對話脈絡。')
            while True:
                try:
                    question = input('你：')
                except EOFError:
                    break
                if normalize(question) in ('exit', 'quit', '離開'):
                    break
                print('機器人：' + bot.reply(question)['answer'])
    except ImportError as exc:
        parser.error('請安裝 requirements-nlp.txt，或使用 --nlp simple：' + str(exc))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
