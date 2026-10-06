"""Dependency-free supervised and lexicon-based news sentiment baselines."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

LABELS = ("negative", "neutral", "positive")
POSITIVE = ("成長", "獲利", "上漲", "突破", "利多", "改善", "創新高", "強勁", "growth", "profit", "gain", "surge", "improve")
NEGATIVE = ("衰退", "虧損", "下跌", "裁員", "利空", "惡化", "暴跌", "違約", "decline", "loss", "fall", "layoff", "crisis", "default")
NEGATIONS = ("不", "未", "沒有", "無", "not", "no", "never")


def tokenize(text):
    """English words plus Chinese character unigrams and bigrams."""
    tokens = re.findall(r"[a-z]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class NaiveBayes:
    def fit(self, rows):
        self.documents = Counter()
        self.counts = {label: Counter() for label in LABELS}
        self.vocabulary = set()
        for row in rows:
            label = row.get("label")
            if label not in LABELS:
                raise ValueError("label 必須為 negative、neutral 或 positive")
            tokens = tokenize(row["text"])
            if not tokens:
                raise ValueError("訓練新聞不可為空或不含可辨識文字")
            self.documents[label] += 1
            self.counts[label].update(tokens)
            self.vocabulary.update(tokens)
        if any(not self.documents[label] for label in LABELS):
            raise ValueError("訓練資料必須包含三種情緒標籤")
        self.totals = {label: sum(count.values()) for label, count in self.counts.items()}
        return self

    def predict(self, text):
        tokens = [token for token in tokenize(text) if token in self.vocabulary]
        if not tokens:
            return {"label": "neutral", "score": 0.0, "probabilities": None, "status": "no_known_tokens"}
        logs = {}
        for label in LABELS:
            logs[label] = math.log(self.documents[label] / sum(self.documents.values()))
            denominator = self.totals[label] + len(self.vocabulary)
            for token in tokens:
                logs[label] += math.log((self.counts[label][token] + 1) / denominator)
        weights = {label: math.exp(value - max(logs.values())) for label, value in logs.items()}
        probabilities = {label: weight / sum(weights.values()) for label, weight in weights.items()}
        return {"label": max(probabilities, key=probabilities.get),
                "score": probabilities["positive"] - probabilities["negative"],
                "probabilities": probabilities, "status": "ok"}


def lexicon_sentiment(text):
    text = text.lower()
    matches = []
    for words, polarity in ((POSITIVE, 1), (NEGATIVE, -1)):
        for word in words:
            pattern = re.escape(word) if re.search(r"[\u4e00-\u9fff]", word) else r"\b" + word + r"\b"
            for match in re.finditer(pattern, text):
                prefix = re.split(r"[，。！？,.!?;；\n]", text[:match.start()])[-1][-12:]
                negated = any(re.search(re.escape(negation) + r"\s*$", prefix) for negation in NEGATIONS)
                matches.append({"term": word, "polarity": -polarity if negated else polarity})
    score = sum(item["polarity"] for item in matches) / len(matches) if matches else 0.0
    return {"label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
            "score": score, "matches": matches, "status": "ok" if matches else "no_lexicon_matches"}


def read_csv(path, training=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"text", "label"} if training else {"text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV 缺少欄位：" + ", ".join(sorted(required)))
        rows = list(reader)
        if any(not row.get("text", "").strip() for row in rows):
            raise ValueError("新聞 text 不可為空")
        return rows


def cluster_news(texts):
    """Unsupervised TF-IDF / spherical k-means, with lexicon interpretation."""
    if not texts:
        return []
    counts = [Counter(tokenize(text)) for text in texts]
    frequency = Counter(token for count in counts for token in count)

    def normalize(vector):
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {key: value / norm for key, value in vector.items()} if norm else {}

    vectors = [normalize({token: value * (1 + math.log((1 + len(texts)) / (1 + frequency[token])))
                          for token, value in count.items()}) for count in counts]

    def similarity(left, right):
        return sum(value * right.get(key, 0) for key, value in left.items())

    centers = [vectors[0]]
    for _ in range(min(3, len(texts)) - 1):
        candidate = min(vectors, key=lambda vector: max(similarity(vector, center) for center in centers))
        if candidate in centers:
            break
        centers.append(candidate)
    assignments = None
    for _ in range(100):
        updated = [max(range(len(centers)), key=lambda index: similarity(vector, centers[index]))
                   for vector in vectors]
        if updated == assignments:
            break
        assignments = updated
        for index in range(len(centers)):
            total = Counter()
            for vector, assignment in zip(vectors, assignments):
                if assignment == index:
                    total.update(vector)
            if total:
                centers[index] = normalize(total)
    scores = [lexicon_sentiment(text)["score"] for text in texts]
    cluster_scores = {}
    for index in range(len(centers)):
        members = [score for score, assignment in zip(scores, assignments) if assignment == index]
        cluster_scores[index] = sum(members) / len(members) if members else 0.0
    return [{"cluster": assignment, "interpreted_label": "positive" if cluster_scores[assignment] > 0
             else "negative" if cluster_scores[assignment] < 0 else "neutral",
             "cluster_lexicon_score": cluster_scores[assignment]}
            for assignment in assignments]


def main():
    parser = argparse.ArgumentParser(description="新聞情緒：Naive Bayes、分群、詞典與可選 Transformer")
    parser.add_argument("--input", required=True, help="含 text 欄位的 UTF-8 CSV")
    parser.add_argument("--train", help="含 text,label 欄位的訓練 CSV")
    parser.add_argument("--output", required=True, help="輸出 JSONL")
    parser.add_argument("--deep-learning", action="store_true", help="啟用預訓練 Transformer 情緒分析")
    parser.add_argument("--deep-model", default="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    parser.add_argument("--device", default="cpu", help="cpu、mps（Apple Silicon）或 cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512, help="包含特殊 token 的最大輸入長度")
    parser.add_argument("--nlp-backend", choices=('auto', 'snownlp', 'vader'), help="啟用 Python NLP 套件情緒分析")
    args = parser.parse_args()
    try:
        rows = read_csv(args.input)
        model = NaiveBayes().fit(read_csv(args.train, training=True)) if args.train else None
        clusters = cluster_news([row["text"] for row in rows])
        deep_results = [None] * len(rows)
        if args.deep_learning:
            from .deep_sentiment import DeepSentiment
            deep_model = DeepSentiment(args.deep_model, args.device, args.batch_size, args.max_length)
            deep_results = deep_model.predict_many([row["text"] for row in rows])
        results = []
        nlp_results = [None] * len(rows)
        if args.nlp_backend:
            from .nlp_sentiment import NLPSentiment
            nlp_results = NLPSentiment(args.nlp_backend).predict_many([row['text'] for row in rows])
        for row, cluster, deep, nlp in zip(rows, clusters, deep_results, nlp_results):
            results.append({**row, "supervised": model.predict(row["text"]) if model else None,
                            "nlp_sentiment": nlp,
                            "deep_learning": deep,
                            "unsupervised_cluster": cluster,
                            "unsupervised_lexicon": lexicon_sentiment(row["text"])})
        with Path(args.output).open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
