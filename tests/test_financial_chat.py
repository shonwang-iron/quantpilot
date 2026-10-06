import unittest
from pathlib import Path

from quantpilot.financial_chat import FinancialChat, convert_twse, read_data, validate_records

ROOT = Path(__file__).resolve().parents[1]


class FinancialChatTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_data(ROOT / 'examples/financial_demo.csv')
        self.bot = FinancialChat(self.rows)

    def test_company_and_followup(self):
        result=self.bot.reply('範例半導體本益比是多少？')
        self.assertEqual(result['facts'][0]['value'],18.5)
        self.assertIn('合成示範資料',result['answer'])
        followup=self.bot.reply('那殖利率呢？')
        self.assertEqual(followup['facts'][0]['metric'],'dividend_yield')
        self.assertEqual(followup['facts'][0]['symbol'],'DEMO1')

    def test_unknown_company_cannot_reuse_previous_company(self):
        self.bot.reply('範例半導體本益比')
        self.assertEqual(self.bot.reply('不在資料中的公司殖利率呢？')['facts'],[])
        self.assertEqual(self.bot.reply('9999 本益比')['facts'],[])

    def test_multiple_companies_missing_ratio_and_definition(self):
        result=self.bot.reply('比較範例半導體與範例電子的本益比')
        self.assertEqual(len(result['facts']),2)
        self.assertEqual(self.bot.reply('範例電子 ROE')['facts'],[])
        self.assertIn('缺少',self.bot.reply('範例電子 ROE')['answer'])
        self.assertEqual(self.bot.reply('什麼是本益比？')['facts'],[])

    def test_history_is_not_answered_with_current_data(self):
        self.assertEqual(self.bot.reply('範例半導體 2025 年本益比')['facts'],[])
        self.assertEqual(self.bot.reply('範例半導體2025年本益比')['facts'],[])
        self.assertIn('歷史',self.bot.reply('範例半導體2025年本益比')['answer'])

    def test_reset_and_english_alias(self):
        self.assertEqual(self.bot.reply('Demo Semiconductor P/E')['facts'][0]['metric'],'pe')
        self.bot.reply('reset')
        self.assertEqual(self.bot.reply('那殖利率呢')['facts'],[])

    def test_official_mapping_and_missing_is_not_zero(self):
        rows=convert_twse([{'Date':'1151005','Code':'2330','Name':'台積電','PEratio':'','PBratio':'2.5','DividendYield':'0'}],'2026-10-06T01:00:00Z')
        self.assertIsNone(rows[0]['value'])
        self.assertEqual(rows[0]['period'],'2026-10-05')
        self.assertEqual(rows[2]['value'],0)

    def test_reject_duplicate_and_invalid_unit(self):
        with self.assertRaises(ValueError):
            validate_records(self.rows+self.rows[:1])
        bad=dict(self.rows[0],unit='%')
        with self.assertRaises(ValueError):
            validate_records([bad])

    def test_long_company_name_does_not_match_substring_company(self):
        short=dict(self.rows[0],symbol='DEMO3',company='範例',aliases='')
        bot=FinancialChat(self.rows+[short])
        self.assertEqual(len(bot.reply('範例半導體與範例電子本益比')['facts']),2)
        self.assertEqual(bot.reply('範例本益比')['facts'][0]['symbol'],'DEMO3')

    def test_ambiguous_alias_requires_clarification(self):
        rows=[dict(self.rows[0],aliases='共同別名'),dict(self.rows[-1],aliases='共同別名')]
        bot=FinancialChat(rows)
        self.assertEqual(bot.reply('共同別名本益比')['facts'],[])
        self.assertIn('代號',bot.reply('共同別名本益比')['answer'])
