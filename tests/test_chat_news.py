import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from quantpilot.chat_news import ChatNews, keyword_match
from quantpilot.financial_chat import FinancialChat, read_data
from quantpilot.news_data import preprocess

ROOT=Path(__file__).resolve().parents[1]
NOW=datetime(2026,10,6,4,tzinfo=timezone.utc)


class ChatNewsTests(unittest.TestCase):
    def setUp(self):
        self.service=ChatNews(ROOT/'examples/chat_news_demo.jsonl')
        self.bot=FinancialChat(read_data(ROOT/'examples/financial_demo.csv'),news_service=self.service)

    def test_industry_summary_and_company_filter(self):
        result=self.service.query(['半導體'],'半導體',now=NOW)
        self.assertEqual(result['sentiment_summary']['articles'],2)
        self.assertEqual(result['sentiment_summary']['positive'],1)
        self.assertEqual(result['sentiment_summary']['negative'],1)
        company=self.service.query(['範例半導體'],'範例半導體',now=NOW)
        self.assertEqual(len(company['news']),1)

    def test_future_news_and_old_publication_are_excluded(self):
        self.service.rows,_=preprocess([
            {'text':'半導體成長','published_at':'2026-09-01T01:00:00Z','available_at':'2026-10-06T01:00:00Z'},
            {'text':'半導體突破','available_at':'2026-10-07T01:00:00Z'}])
        self.assertEqual(self.service.query(['半導體'],'半導體',now=NOW)['news'],[])

    def test_company_followup_unknown_and_reset(self):
        with patch('quantpilot.chat_news.datetime') as clock:
            clock.now.return_value=NOW
            self.assertEqual(len(self.bot.reply('範例半導體新聞')['news']),1)
            self.assertEqual(len(self.bot.reply('那情緒如何？')['news']),1)
            self.assertEqual(self.bot.reply('未知公司新聞呢')['facts'],[])
            self.assertNotIn('news',self.bot.reply('未知公司新聞呢'))
            self.bot.reply('重設')
            self.assertNotIn('news',self.bot.reply('那情緒如何？'))

    def test_industry_context_and_no_multiple_company_mix(self):
        with patch('quantpilot.chat_news.datetime') as clock:
            clock.now.return_value=NOW
            self.assertEqual(len(self.bot.reply('半導體產業新聞')['news']),2)
            self.assertEqual(len(self.bot.reply('那情緒如何？')['news']),2)
            self.assertNotIn('news',self.bot.reply('範例半導體和範例電子新聞'))

    def test_refresh_failure_is_not_silently_cached_success(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'sources.json'
            source.write_text('[{"name":"test","url":"https://example.com/feed"}]')
            service=ChatNews(ROOT/'examples/chat_news_demo.jsonl',source,Path(folder)/'raw.jsonl')
            with patch('quantpilot.chat_news.collect',return_value=([],[{'source':'test','status':'error'}])):
                with self.assertRaises(RuntimeError):
                    service.query(['半導體'],'半導體',refresh=True,now=NOW)

    def test_english_keyword_boundaries_and_config_validation(self):
        self.assertFalse(keyword_match('chair','AI'))
        self.assertTrue(keyword_match('AI chip','AI'))
        with self.assertRaises(ValueError):
            ChatNews(limit=0)

    def test_price_number_is_not_company_code(self):
        rows=[dict(read_data(ROOT/'examples/financial_demo.csv')[0],symbol='2330',company='測試晶圓',aliases='')]
        service=ChatNews()
        service.rows,_=preprocess([
            {'text':'其他公司股價2330元上漲','available_at':'2026-10-06T01:00:00Z'},
            {'text':'測試晶圓（2330）獲利成長','available_at':'2026-10-06T01:01:00Z'}])
        bot=FinancialChat(rows,news_service=service)
        with patch('quantpilot.chat_news.datetime') as clock:
            clock.now.return_value=NOW
            result=bot.reply('2330新聞')
        self.assertEqual(len(result['news']),1)
        self.assertIn('測試晶圓',result['news'][0]['text'])


if __name__=='__main__':
    unittest.main()
