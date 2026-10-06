# quantpilot

新聞情緒分析，支援中文與英文。基準方法只需 Python 3 標準函式庫；可選的深度學習模型需要 PyTorch 與 Transformers。

- **監督式**：Multinomial Naive Bayes，以人工標註新聞訓練，輸出 negative / neutral / positive、類別機率及情緒分數。
- **非監督式學習**：TF-IDF 與 spherical k-means，依新聞文字相似度分成最多三群，再以群內詞典情緒平均解讀群組。群組可能反映主題而非情緒；正負標籤來自詞典解讀，不是分群自行學到的語義。
- **非監督／免標註基準**：情緒詞典計分，輸出情緒、分數及命中詞。這是規則式方法，不是從未標註資料學習的分群模型；分群本身也不能直接決定正負情緒。

準備 UTF-8 CSV，新聞檔包含 `text`，訓練檔包含 `text,label`，標籤必須涵蓋三種類別。其他新聞欄位（例如 id、日期、來源）會保留。

```csv
text,label
公司獲利成長,positive
公司虧損惡化,negative
公司公布股東會日期,neutral
```

## 執行前準備

安裝 Python 3，確認終端機可以執行 Python。基準分析與比較工具只使用標準函式庫，不需要第三方套件；啟用深度學習時請依下方說明安裝。

將 `news.csv` 與 `labeled_news.csv` 放在專案根目錄（與 README.md 同一層）。`news.csv` 範例：

```csv
text
公司獲利成長
公司虧損惡化
公司公布股東會日期
```

`labeled_news.csv` 使用上方的 `text,label` 格式。CSV 請儲存為 UTF-8 或 UTF-8 BOM。以下指令必須在專案根目錄執行；輸出檔案若已存在會被覆寫。

## macOS（Terminal）

開啟 Terminal，進入專案並確認 Python 版本：

```sh
cd "/Users/shonwang/Documents/Self/Code/quantpilot"
python3 --version
```

同時執行監督式、非監督式分群及詞典分析：

```sh
python3 -m quantpilot.sentiment --input news.csv --train labeled_news.csv --output sentiment.jsonl
```

沒有標註資料時，僅執行分群與詞典分析，`supervised` 輸出 `null`：

```sh
python3 -m quantpilot.sentiment --input news.csv --output sentiment.jsonl
```

執行測試及查看結果：

```sh
python3 -m unittest discover -s tests
cat sentiment.jsonl
```

## Windows（PowerShell）

開啟 PowerShell，將下方路徑替換成 Windows 上的實際專案路徑：

```powershell
Set-Location "C:\Users\你的使用者名稱\Documents\Self\Code\quantpilot"
py -3 --version
```

同時執行監督式、非監督式分群及詞典分析：

```powershell
py -3 -m quantpilot.sentiment --input news.csv --train labeled_news.csv --output sentiment.jsonl
```

沒有標註資料時，僅執行分群與詞典分析，`supervised` 輸出 `null`：

```powershell
py -3 -m quantpilot.sentiment --input news.csv --output sentiment.jsonl
```

執行測試及以 UTF-8 查看結果：

```powershell
py -3 -m unittest discover -s tests
Get-Content -Encoding UTF8 sentiment.jsonl
```

若無法辨識 `py`，但 `python --version` 顯示 Python 3，可將上述 `py -3` 改為 `python`。若兩者皆無法執行，請先安裝 Python 3 並確認其命令可在終端機使用，再重新開啟 PowerShell。

## 常見問題

- `No module named quantpilot`：確認目前目錄為專案根目錄，且其下有 `quantpilot` 資料夾。
- 找不到 CSV：確認檔名與所在目錄，或在 `--input`、`--train` 後指定完整路徑；含空白的路徑需加雙引號。
- 訓練資料缺少類別：`label` 必須包含 `negative`、`neutral`、`positive` 三種類別，且每類至少一筆有效新聞。
- 只想確認指令參數：macOS 執行 `python3 -m quantpilot.sentiment --help`，Windows 執行 `py -3 -m quantpilot.sentiment --help`。

## 輸出與限制

輸出每行一則 JSON，`supervised` 與 `unsupervised_lexicon` 分別保留結果。分數範圍為 -1 到 1：Naive Bayes 使用 P(positive) − P(negative)，詞典使用命中詞極性平均。兩種分數不可視為同一尺度的信心水準。無已知詞或無詞典命中時回傳 neutral 與明確 status，表示缺乏證據。

`unsupervised_cluster` 包含群組編號、詞典解讀標籤及群內平均分數。分群對整批輸入重新擬合，群組編號在不同批次之間不可比較；使用確定性的最遠點初始化與最多 100 次迭代，適合小型基準分析。

這些是可解釋的基準模型，尚未用真實新聞驗證。正式使用前應準備獨立人工標註測試集，以時間切分並去除重複新聞，檢查 macro-F1、各類別 precision/recall 及混淆矩陣。Naive Bayes 機率未校準；詞典、簡單否定規則及中文字元切詞不能可靠處理反諷、複雜否定或事件對不同公司的影響。上面的三筆資料僅說明格式，不足以訓練實用模型。


## 深度學習情緒分析

使用預訓練 Transformer，依上下文產生三類情緒機率。預設為 [多語言 DistilBERT](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)，中文新聞可先使用此模型作為深度學習基準；它不是針對台灣財經新聞微調的模型，繁體中文效果需要實際驗證。英文財經新聞可指定 [FinBERT](https://huggingface.co/ProsusAI/finbert)。推論介面依據 [Transformers 官方文件](https://huggingface.co/docs/transformers/main_classes/pipelines)。

建議使用 Python 3.11 或 3.12 建立獨立環境。以下指令均在專案根目錄執行。

### macOS

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-deep.txt
.venv/bin/python -m quantpilot.sentiment --input news.csv --deep-learning --output sentiment.jsonl
.venv/bin/python -m unittest discover -s tests
```

Apple Silicon 若 PyTorch 支援 MPS，可在分析指令加上 `--device mps`；預設 CPU 不要求 GPU。

### Windows（PowerShell）

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deep.txt
.\.venv\Scripts\python.exe -m quantpilot.sentiment --input news.csv --deep-learning --output sentiment.jsonl
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

直接使用虛擬環境中的 Python，無須啟用 PowerShell 啟用腳本。使用 NVIDIA GPU 時，須先安裝適用的 CUDA 版 PyTorch，再指定 `--device cuda:0`。

### 選項與結果

- `--deep-learning`：啟用模型。首次執行會從 Hugging Face 下載模型與 tokenizer；後續可使用本機快取。新聞推論在本機執行。
- `--deep-model ProsusAI/finbert`：切換英文財經模型，也可指定相容的本機模型目錄。模型的 `id2label` 必須明確對應三種情緒名稱。
- `--batch-size 8`：每批新聞數量；記憶體不足時可降至 1。
- `--max-length 512`：最大 token 數，包含特殊 token，且不超過模型支援上限。超過長度只分析前段，輸出 `status: truncated`。完整長文需另行切段與驗證聚合方式。
- 可同時加上 `--train labeled_news.csv`，比較 Naive Bayes 與深度學習輸出。`--train` 只訓練 Naive Bayes，不會微調 Transformer。

新增的 `deep_learning` 欄位包含 `label`、`probabilities`、`confidence`、`score`（P(positive) − P(negative)）、模型名稱、原始 token 數與截斷狀態。未啟用時為 `null`。機率與 confidence 未校準，不能當成準確率；程式也不會將模型錯誤偽裝成 neutral。

此整合使用現有模型推論，尚未以專案新聞資料驗證準確率提升。正式比較請使用同一份獨立人工標註測試集，計算 macro-F1 與各類別 precision/recall；若要提高特定財經領域的效果，後續需要足量標註資料進行微調。單元測試使用模擬模型驗證整合邏輯，不代表已執行真實模型推論。


## 台股情緒方法比較與策略回測

| 方法 | 優點 | 比較時須注意 | 策略研究用途 |
| --- | --- | --- | --- |
| 情緒詞典 | 快速、可解釋、免訓練 | 否定、語境與新詞可能誤判 | 作為成本低的基準 |
| Naive Bayes | 可用領域標註資料訓練 | 依賴標註品質，弱於複雜語境 | 檢查領域標註是否有幫助 |
| TF-IDF + K-means | 可觀察新聞群組 | 群組通常可能反映主題；情緒仍依賴詞典 | 作為探索性對照，不是獨立情緒真值 |
| 多語言 DistilBERT | 可使用上下文 | 繁體中文財經需驗證；長文可能截斷 | 與基準比較是否改善分類與交易績效 |
| FinBERT | 英文財經領域模型 | 不能直接當成台股中文模型 | 有英文新聞時另行比較 |

### 資料與時間規則

每次比較一個標的，請先篩選與該公司相關的新聞。`--news` CSV 需有 `text,available_at`；`label` 可選，提供後才能計算分類指標。`available_at` 是實際取得新聞、可產生訊號的時間，必須含時區（台股使用 `+08:00`）。資料依時間排序，完全相同的文字只保留最早一筆；近似轉載仍需自行清理。

`--train` CSV 需有 `text,available_at,label`。所有訓練新聞必須早於第一筆進場時間，且不能與評估新聞文字重複。訓練標註也必須在回測起點前可取得。模型於回測期間固定，不會拿未來標註重訓。

`--periods` CSV 指定實際可成交的進出場時間與價格：

```csv
symbol,entry_at,exit_at,entry_price,exit_price
DEMO,2026-01-06T09:01:00+08:00,2026-01-07T09:01:00+08:00,100,102
```

價格請使用與進出場時點一致的資料，並一致處理除權息、拆股與股利；本工具不自動調整。隔日交易可用次日同一時點出場，持有數日則延後出場時間。交易期間不可重疊；一個標的每期至多一筆持倉。範例時間與價格都是合成資料。

只有嚴格早於進場的新聞可產生訊號；每筆新聞在第一次可用的交易期間使用一次。同期間多則新聞取平均情緒分數。請讓評估新聞從研究起始日開始，避免第一期混入過久的舊新聞；本版沒有自動新聞衰減。分群在每期只使用當時已取得的歷史新聞重新擬合，不使用 `sentiment.jsonl` 中整批擬合的群組，避免未來資訊洩漏。

### 執行比較

以下範例純粹驗證流程，不是投資績效證據。`examples/` 中的新聞、標註及價格均為合成資料。

macOS：

```sh
python3 -m quantpilot.compare --news examples/news_demo.csv --train examples/train_demo.csv --periods examples/periods_demo.csv --threshold 0.2 --cost-bps 10 --output comparison.json
```

Windows PowerShell：

```powershell
py -3 -m quantpilot.compare --news examples/news_demo.csv --train examples/train_demo.csv --periods examples/periods_demo.csv --threshold 0.2 --cost-bps 10 --output comparison.json
```

加入深度學習比較（先安裝 requirements-deep.txt；Windows 改用虛擬環境 Python）：

```sh
.venv/bin/python -m quantpilot.compare --news news_eval.csv --train train_history.csv --periods periods.csv --deep-model lxyuan/distilbert-base-multilingual-cased-sentiments-student --threshold 0.2 --cost-bps 10 --output comparison.json
```

`--deep-model` 可重複指定以比較多個三類情緒模型；請確認每個模型適用於輸入新聞語言。若沒有訓練資料可省略 `--train`，結果不包含 Naive Bayes。

### 策略與績效解讀

預設做多／空手：平均情緒分數大於 threshold 時做多，其餘空手；無新聞也空手。每期進場與出場後歸零，下一期重新判斷；資金逐期複利。`--allow-short` 才啟用負面訊號放空，但未模擬台股融券可用性、借券費、漲跌停與成交限制。

`--cost-bps` 是每邊有效成本，每筆完整交易扣兩次，涵蓋自訂手續費、滑價與交易稅的近似值。若買賣成本不對稱，可把完整來回成本除以二作為此參數。10 bps 只是流程範例，不是台股適用成本建議；請依交易標的、券商及交易方式設定。工具假設可按指定價格完整成交，不含容量、市場衝擊、隔夜跳空成交限制與閒置資金利息。

`comparison.json` 提供：

- 人工標註對照：accuracy、macro-F1、每類 precision/recall/F1、support 及混淆矩陣；缺少人工標註時為 null。
- 方法間的情緒標籤一致率，以及每期訊號、部位、成本前後報酬。
- 成本前後總報酬、期末資金最大回撤、完整交易數、勝率及交易期間占比。
- 同期間每期做多並完整出場的基準與空手基準。做多基準不是連續買入持有；區間外報酬不計入。

`untruncated_ok_news` 表示狀態為 ok 的新聞數，並非準確率；分群的情緒解讀仍來自詞典。最大回撤只觀察各期間結束時的資金，無法顯示持倉期間內的回撤。不計算年化 Sharpe，因為使用者提供的期間可能不等長。

各方法的分數尺度不同，先用固定門檻對照，再於獨立驗證期間分別選定門檻，最後用未參與選擇的測試期間比較。不要在同一段測試資料反覆調參後挑最高報酬；分類 F1 高也不代表可獲利。時間順序切分的原則見 [scikit-learn 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。此工具檢查資料時間，但無法驗證新聞的真實首次取得時間或預訓練模型的歷史可用性；使用回測當時尚未發布的模型，結果只能視為回溯研究。

目前未提供真實台股回測結論。應以獨立期間的扣成本報酬、回撤、交易數及不同市場階段穩定性決定是否值得繼續研究。回測為假設績效，與實際交易表現不同，參考 [SEC 投資人績效說明](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-47)。


## 情緒與詞彙向量作為交易特徵

新增 `quantpilot.feature_strategy`：使用 Ridge 正則化迴歸預測每個持有期間的價格報酬，不把正面情緒直接等同上漲。每期彙整在進場前已取得且尚未使用的新聞，建立以下特徵：

- 情緒平均值、情緒分散程度、正面／負面／中性新聞比例。
- `log(1 + 新聞數)` 與有效且未截斷的情緒結果比例。
- TF-IDF 詞彙向量，使用現有英文詞及中文字元 unigram／bigram 切詞，預設最多 2,000 維。這是稀疏詞彙向量，尚未加入 Word2Vec 或 Transformer 語意 embedding。

預設情緒由詞典產生；可用 `--deep-model` 改為 Transformer，並安裝 requirements-deep.txt。此流程不使用人工新聞 label 作為交易特徵，也不使用未來價格作為輸入特徵。迴歸目標為 `exit_price / entry_price - 1`，僅供監督式訓練及事後評估。

### 安裝與執行

在專案根目錄執行。新聞與價格使用前述 `available_at` 及進出場 CSV 格式，資料應同時涵蓋訓練與後續測試期間；所有時間必須包含時區，且一次只研究一個標的。以下資料為合成範例，12 個期間中後 6 個用來測試；日期與價格不代表真實台股交易。

macOS：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-strategy.txt
.venv/bin/python -m quantpilot.feature_strategy --news examples/feature_news_demo.csv --periods examples/feature_periods_demo.csv --test-start 2026-01-23T09:01:00+08:00 --min-train 3 --output feature_comparison.json
.venv/bin/python -m unittest discover -s tests
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-strategy.txt
.\.venv\Scripts\python.exe -m quantpilot.feature_strategy --news examples/feature_news_demo.csv --periods examples/feature_periods_demo.csv --test-start 2026-01-23T09:01:00+08:00 --min-train 3 --output feature_comparison.json
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

正式研究請替換為歷史資料，省略 `--min-train 3` 以使用預設最少 30 個歷史期間；30 只是最低樣本門檻，並非足以支持 2,000 維模型或交易結論的保證。

### 逐期訓練與消融比較

每個測試進場時點，使用出場時間**嚴格早於進場**的全部歷史期間重新訓練（expanding window）。TF-IDF 詞彙、IDF、情緒特徵標準化與 Ridge 係數均只使用當時的訓練資料。後續測試期間已出場的結果可以用於下一期重訓，這是逐期更新的策略，不是固定模型留出測試。前處理與資料洩漏原則見 [scikit-learn 官方文件](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。

報告在相同測試期間比較六組策略：

| 名稱 | 輸入與行為 |
| --- | --- |
| sentiment_only | 只用七個情緒／新聞統計特徵 |
| words_only | 只用 TF-IDF 詞彙向量 |
| combined | 合併情緒統計與 TF-IDF |
| historical_mean | 用已完成歷史期間平均報酬預測 |
| long_every_period | 每個評估期間做多並完整出場 |
| cash | 全程空手 |

前三組採相同 Ridge 設定，藉此檢查合併特徵是否比單組特徵提供更好的樣本外預測與交易績效。合併不一定較好；高維詞彙特徵可能過擬合。

### 決策與可設定參數

預設做多／空手，條件為 `預測報酬 > 2 × cost-bps / 10000 + edge`。`--edge` 預設 0.001（額外要求 0.1% 預測淨報酬），`--cost-bps` 預設 10 僅為示範，台股成本需自行設定。無新新聞時，四組預測策略皆空手。

- `--alpha 1`：Ridge 正則化強度，必須大於零。
- `--max-features 2000`：TF-IDF 維度上限；資料較少時可降低。
- `--min-train 30`：最少已完成歷史期間；不足的測試期會跳過並記錄。
- `--test-start`：含時區的測試起點。
- `--deep-model lxyuan/distilbert-base-multilingual-cased-sentiments-student`：改用深度情緒特徵。

報告包含預測 MAE／RMSE、扣成本前後報酬、最大回撤、交易數、勝率、每期特徵、預測值、訓練筆數、詞彙數及最近訓練出場時間。向量依每期訓練詞彙表重新建立；報告不展開所有稀疏向量。無有效文字的訓練資料使用零詞彙特徵及截距基準。

調整 alpha、詞彙維度、edge 與成本前，先保留獨立驗證期間，再鎖定設定測試後續期間。未加入超參數搜尋、成交限制、新聞時間衰減、持倉期間內價格路徑、保存模型或即時下單；交易成本與期間邊界假設與前述回測一致。預訓練模型在歷史上的可用性、新聞與價格品質仍需自行驗證。

單元測試以真實 scikit-learn 迴歸驗證整合、訓練時序、未來資料不影響過去決策與成本。範例僅驗證流程，尚無真實台股資料證明合併特徵提升收益。


## 多來源新聞收集與情緒分析前處理

新增 `quantpilot.news_data`，只使用 Python 標準函式庫。支援 RSS／Atom 收集及 CSV／JSONL 匯入。預設來源設定位於 `examples/rss_sources.json`：

- [中央社 RSS 服務](https://www.cna.com.tw/about/rss.aspx)：產經證券，`https://feeds.feedburner.com/rsscna/finance`。
- [Yahoo 股市 RSS 服務](https://tw.stock.yahoo.com/rss-index)：台股動態，`https://tw.stock.yahoo.com/rss?category=tw-market`。

RSS 主要包含標題、摘要與文章連結，本程式不抓取文章全文。這兩個官方 RSS 的免費使用規範限私人／非商業用途，請保留來源；商業策略用途須確認授權。來源條款見中央社上述服務頁及 [Yahoo 使用說明](https://tw.stock.yahoo.com/rss-help)。設定檔可替換為已取得授權的 HTTPS RSS／Atom。

### 收集、清理、分析

請先建立 data 資料夾。collect 會**追加**到原始 JSONL，而 prepare 會讀取累積資料、重新產生去重後的 CSV。重跑收集可保留首次取得時間，不需另外的第三方套件。

macOS：

```sh
mkdir -p data
python3 -m quantpilot.news_data collect --sources examples/rss_sources.json --raw-output data/news_raw.jsonl --report data/collection_report.json
python3 -m quantpilot.news_data prepare --input data/news_raw.jsonl --output data/news_clean.csv --report data/quality_report.json
python3 -m quantpilot.sentiment --input data/news_clean.csv --output data/news_sentiment.jsonl
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force data
py -3 -m quantpilot.news_data collect --sources examples/rss_sources.json --raw-output data/news_raw.jsonl --report data/collection_report.json
py -3 -m quantpilot.news_data prepare --input data/news_raw.jsonl --output data/news_clean.csv --report data/quality_report.json
py -3 -m quantpilot.sentiment --input data/news_clean.csv --output data/news_sentiment.jsonl
```

需要深度情緒時，安裝 requirements-deep.txt，以虛擬環境 Python 執行最後一個指令並加 `--deep-learning`。本次實際收集的 data/news_sentiment.jsonl 僅執行詞典與分群，supervised 與 deep_learning 為 null；新聞尚無人工情緒標註。

collect 的來源報告列出各來源成功／失敗、筆數與取得時間；來源失敗不會丟失其他來源的成功資料，但指令以非零狀態結束，方便偵測缺失。每個來源有 20 秒 timeout 與 5 MiB 上限。若整批來源都失敗，本次不新增新聞；舊 raw 資料仍保留，不應誤認為本次的新資料。尚未加入排程、API 金鑰來源或重試機制。

### 匯入與篩選

可重複指定 input 合併不同的 CSV／JSONL：

```sh
python3 -m quantpilot.news_data prepare --input data/news_raw.jsonl --input vendor_news.csv --keyword 台積電 --keyword TSMC --output data/tsmc_news.csv --report data/tsmc_quality.json
```

多個 keyword 採任一命中；這是文字篩選，不是公司實體辨識，仍需人工檢查關聯性。匯入資料必須有帶時區的 `available_at`，及 `text` 或 `title`／`summary`。建議另附 `source,url,published_at,label`；label 可省略。available_at 必須是可信的實際取得時間，程式不會用發布時間補填或替使用者推定歷史可用時間。

### 前處理與品質紀錄

- 清除 HTML 標籤、script／style、控制字元與多餘空白，進行 Unicode NFKC（如全形數字與百分比）正規化。保留否定詞、標點、數字、百分比與中英混合內容，不自動翻譯或轉換繁簡體。原始文字保留在 raw JSONL。
- RSS／Atom 發布時間統一為含時區的 UTC；缺少或無法解析則留空並標記。available_at 是每個來源完整讀取後的實際時間；發布晚於取得的資料進入品質報告，不輸出到分析 CSV。
- 依清理後文字 casefold 雜湊做完全去重，保留最早 available_at 與當時的 title、summary、url、label。不同來源的相同內容可合併；不同摘要或近似轉載仍會保留。URL 移除 utm、fbclid、gclid 與 fragment，但保留文章識別參數；相同 URL 的不同文字視為不同版本。
- 保留來源、連結及資料 ID；sources 記錄所有重複來源，僅供追溯，包含可能較晚才取得的資訊，**不可直接作為過去策略的來源數特徵**。
- 空文字、短於 min-chars（預設 4）、無有效取得時間、無效 label、未命中 keyword 等記錄，列入報告並附原始索引。沒有命中不等於沒有新聞，需看報告原因。
- 重複內容的人工標籤若互相衝突，清空 label 並標記 conflicting_labels，需人工處理後才可用於監督式訓練。不以未來取得的標籤補填較早記錄。

news_clean.csv 可直接供 sentiment、compare 及 feature_strategy 使用。資料時間採 UTC ISO 8601，分析工具會依時區正確比較；Excel 可讀取 UTF-8 BOM CSV。本工具不會自動移除來源署名、造成人工標註、推定缺失時間、擷取股價或把今日抓取的歷史文章當成過去已取得的新聞。

首次實際驗證收集中央社 20 則及 Yahoo 股市 50 則，共 70 則，清理後保留 69 則（完全重複 1 則、拒絕 0 則）。這是一次收集快照，不是完整歷史資料，也不是單一公司的篩選資料；請先依標的篩選，並累積有可信時間的新聞及相應價格，再進行台股回測。資料位於 data/，已排除於 Git 追蹤。
