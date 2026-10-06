import unittest

from quantpilot.news_data import canonical_url, clean_text, parse_feed, preprocess


class NewsDataTests(unittest.TestCase):
    def test_cleanup_preserves_negation_numbers(self):
        self.assertEqual(clean_text('<p>公司未成長&nbsp;１０％</p><script>bad</script>'), '公司未成長 10%')
        self.assertEqual(clean_text('獲利 < 10%'), '獲利 < 10%')

    def test_rss_and_atom_dates(self):
        rss=b'<rss><channel><item><title>Profit</title><description>Growth</description><pubDate>Tue, 06 Oct 2026 01:00:00 GMT</pubDate></item></channel></rss>'
        rows=parse_feed(rss,'source','2026-10-06T02:00:00+00:00')
        self.assertEqual(rows[0]['published_at'],'2026-10-06T01:00:00+00:00')
        atom=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Profit</title><link href="https://example.com/a"/><published>2026-10-06T01:00:00Z</published></entry></feed>'
        self.assertEqual(parse_feed(atom,'s','2026-10-06T02:00:00Z')[0]['url'],'https://example.com/a')

    def test_dedup_preserves_earliest_available(self):
        raw=[{'text':'公司未成長10%', 'source':'A','available_at':'2026-10-06T02:00:00Z'},
             {'text':'公司未成長１０％','source':'B','available_at':'2026-10-06T01:00:00Z'}]
        rows,report=preprocess(raw)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['source'],'B')
        self.assertEqual(rows[0]['available_at'],'2026-10-06T01:00:00+00:00')
        self.assertEqual(rows[0]['sources'],['A','B'])
        self.assertEqual(report['duplicates_removed'],1)

    def test_missing_time_future_publication_and_keyword(self):
        raw=[{'text':'獲利成長'}, {'text':'獲利成長','available_at':'2026-10-06T01:00:00Z','published_at':'2026-10-06T02:00:00Z'},
             {'text':'公司虧損惡化','available_at':'2026-10-06T01:00:00Z'}]
        rows,report=preprocess(raw,keywords=['成長'])
        self.assertEqual(rows,[])
        self.assertEqual(len(report['rejected']),3)

    def test_label_conflict_is_not_silently_used(self):
        raw=[{'text':'公司發布新公告','available_at':'2026-10-06T01:00:00Z','label':label} for label in ('positive','negative')]
        rows,report=preprocess(raw)
        self.assertEqual(rows[0]['label'],'')
        self.assertEqual(report['label_conflicts'],1)

    def test_url_keeps_meaningful_parameters_and_rejects_dtd(self):
        self.assertEqual(canonical_url('https://EXAMPLE.com/a?id=7&utm_source=x#top'),'https://example.com/a?id=7')
        with self.assertRaises(ValueError):
            parse_feed(b'<!DOCTYPE rss><rss/>','s','2026-10-06T01:00:00Z')


if __name__ == '__main__':
    unittest.main()
