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

安裝 Python 3，確認終端機可以執行 Python。本專案只使用標準函式庫，不需要安裝第三方套件或建立虛擬環境。

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
