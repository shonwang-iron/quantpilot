"""RSS/Atom collection, auditable timestamps, and conservative text cleaning."""

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .compare import timestamp

MAX_FEED_BYTES = 5 * 1024 * 1024
FIELDS = ['id', 'text', 'title', 'summary', 'source', 'url', 'published_at',
          'available_at', 'label', 'sources', 'quality_flags']


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag in ('p', 'br', 'div', 'li'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1
        if tag in ('p', 'div', 'li'):
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    # RSS descriptions can contain escaped HTML, but normal comparison signs remain text.
    value = html.unescape(str(value or ''))
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    value = unicodedata.normalize('NFKC', ''.join(parser.parts))
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]', '', value)
    return re.sub(r'\s+', ' ', value).strip()


def canonical_url(value):
    if not value:
        return ''
    parts = urlsplit(value.strip())
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith('utm_') and key.lower() not in ('fbclid', 'gclid')]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), ''))


def feed_date(value):
    if not value:
        return ''
    try:
        try:
            result = timestamp(value)
        except ValueError:
            result = parsedate_to_datetime(value)
        if result.tzinfo is None:
            return ''
        return result.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return ''


def local_name(tag):
    return tag.rsplit('}', 1)[-1]


def child_text(node, names):
    for name in names:
        for child in node:
            if local_name(child.tag) == name:
                return ''.join(child.itertext())
    return ''


def parse_feed(payload, source, obtained_at):
    if len(payload) > MAX_FEED_BYTES:
        raise ValueError('RSS 超過 5 MiB 上限')
    if b'\x00' in payload or re.search(br'<!\s*(DOCTYPE|ENTITY)\b', payload, re.I):
        raise ValueError('不接受 XML DTD 或 entity 宣告')
    root = ET.fromstring(payload)
    if local_name(root.tag) not in ('rss', 'feed', 'RDF'):
        raise ValueError('來源未回傳 RSS／Atom')
    timestamp(obtained_at)
    records = []
    for item in root.iter():
        if local_name(item.tag) not in ('item', 'entry'):
            continue
        link = child_text(item, ('link',))
        for child in item:
            if local_name(child.tag) == 'link' and child.attrib.get('rel', 'alternate') == 'alternate' and child.attrib.get('href'):
                link = child.attrib['href']
                break
        raw_date = child_text(item, ('pubDate', 'published', 'date', 'updated'))
        records.append({'title': child_text(item, ('title',)),
                        'summary': child_text(item, ('description', 'summary', 'content', 'encoded')),
                        'source': source, 'url': link, 'published_at': feed_date(raw_date),
                        'published_at_raw': raw_date, 'available_at': obtained_at,
                        'collection_type': 'rss'})
    return records


def collect(config):
    records, statuses = [], []
    for source in config:
        try:
            if urlsplit(source['url']).scheme != 'https':
                raise ValueError('RSS 來源必須為 HTTPS')
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'Quantpilot/0.1 (RSS research client)'})
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(MAX_FEED_BYTES + 1)
            obtained = datetime.now(timezone.utc).isoformat()
            rows = parse_feed(payload, source['name'], obtained)
            records.extend(rows)
            statuses.append({'source': source['name'], 'status': 'ok', 'items': len(rows), 'obtained_at': obtained})
        except (OSError, ValueError, ET.ParseError, KeyError) as exc:
            statuses.append({'source': source.get('name', 'unknown'), 'status': 'error', 'error': str(exc)})
    return records, statuses


def read_records(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if path.suffix.lower() == '.csv':
            return list(csv.DictReader(stream))
        return [json.loads(line) for line in stream if line.strip()]


def preprocess(records, min_chars=4, keywords=None):
    if min_chars < 1:
        raise ValueError('min-chars 必須大於 0')
    accepted, rejected = [], []
    for index, raw in enumerate(records):
        try:
            available = timestamp(raw['available_at'])
            title = clean_text(raw.get('title'))
            summary = clean_text(raw.get('summary'))
            text = clean_text(raw.get('text')) or (title + (' ' + summary if summary and summary != title else '')).strip()
            if len(text) < min_chars:
                raise ValueError('文字過短或為空')
            if keywords and not any(keyword.casefold() in text.casefold() for keyword in keywords):
                raise ValueError('不符合關鍵字篩選')
            published = raw.get('published_at', '')
            flags = []
            if published:
                published_time = timestamp(published)
                if published_time > available:
                    raise ValueError('發布時間晚於實際取得時間')
                published = published_time.astimezone(timezone.utc).isoformat()
            else:
                flags.append('missing_or_invalid_published_at')
            label = raw.get('label', '')
            if label and label not in ('positive', 'neutral', 'negative'):
                raise ValueError('label 無效')
            source = clean_text(raw.get('source')) or 'imported'
            digest = hashlib.sha256(text.casefold().encode('utf-8')).hexdigest()
            accepted.append({'id': digest, 'text': text, 'title': title, 'summary': summary,
                             'source': source, 'url': canonical_url(raw.get('url', '')),
                             'published_at': published, 'available_at': available.astimezone(timezone.utc).isoformat(),
                             'label': label, 'sources': [source], 'quality_flags': flags})
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({'record_index': index, 'reason': str(exc)})
    deduplicated = {}
    conflicts = set()
    duplicates = 0
    for row in sorted(accepted, key=lambda item: timestamp(item['available_at'])):
        if row['id'] not in deduplicated:
            deduplicated[row['id']] = row
            continue
        duplicates += 1
        first = deduplicated[row['id']]
        first['sources'] = sorted(set(first['sources'] + row['sources']))
        if first['label'] != row['label'] and first['label'] and row['label']:
            conflicts.add(row['id'])
    # Never silently train on conflicting annotations.
    for key in conflicts:
        deduplicated[key]['label'] = ''
        deduplicated[key]['quality_flags'].append('conflicting_labels')
    return list(deduplicated.values()), {'input_records': len(records), 'accepted_unique': len(deduplicated),
        'duplicates_removed': duplicates, 'rejected': rejected, 'label_conflicts': len(conflicts)}


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'sources': json.dumps(row['sources'], ensure_ascii=False),
                             'quality_flags': json.dumps(row['quality_flags'], ensure_ascii=False)})


def main():
    parser = argparse.ArgumentParser(description='多來源新聞收集及情緒分析前處理')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('collect')
    fetch.add_argument('--sources', required=True, help='來源設定 JSON：name 與 url')
    fetch.add_argument('--raw-output', required=True, help='追加原始記錄的 JSONL')
    fetch.add_argument('--report', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--input', action='append', required=True, help='可重複指定 CSV／JSONL')
    prep.add_argument('--output', required=True)
    prep.add_argument('--report', required=True)
    prep.add_argument('--min-chars', type=int, default=4)
    prep.add_argument('--keyword', action='append', default=[])
    args = parser.parse_args()
    try:
        if args.command == 'collect':
            config = json.loads(Path(args.sources).read_text(encoding='utf-8'))
            if not isinstance(config, list) or not config:
                raise ValueError('來源設定必須為非空陣列')
            if any(not isinstance(item, dict) or not item.get('name') or not item.get('url') for item in config):
                raise ValueError('每個來源需包含 name 與 url')
            rows, statuses = collect(config)
            report = {'sources': statuses, 'collected_records': len(rows)}
            with Path(args.raw_output).open('a', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            records = [row for path in args.input for row in read_records(path)]
            rows, report = preprocess(records, args.min_chars, args.keyword)
            write_csv(args.output, rows)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.command == 'collect' and any(item['status'] != 'ok' for item in statuses):
            parser.exit(1, '部分來源收集失敗，已保存成功資料，請查看來源報告。\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
