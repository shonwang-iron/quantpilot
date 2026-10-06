"""Optional pretrained Transformer inference; no imports until requested."""

import math

DEFAULT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABELS = {"negative", "neutral", "positive"}


class DeepSentiment:
    def __init__(self, model=DEFAULT_MODEL, device="cpu", batch_size=8, max_length=512):
        if batch_size < 1 or max_length < 8:
            raise ValueError("batch-size 必須大於 0，max-length 必須至少為 8")
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("請先安裝深度學習套件：python -m pip install -r requirements-deep.txt") from exc
        self.model_name = model
        self.batch_size = batch_size
        self.classifier = pipeline("text-classification", model=model, tokenizer=model,
                                   device=device, framework="pt", trust_remote_code=False)
        configured_labels = {str(label).lower() for label in self.classifier.model.config.id2label.values()}
        if configured_labels != LABELS:
            raise ValueError("模型 id2label 必須明確對應 negative、neutral、positive 三種類別")
        limits = [max_length]
        for limit in (getattr(self.classifier.tokenizer, "model_max_length", None),
                      getattr(self.classifier.model.config, "max_position_embeddings", None)):
            if isinstance(limit, int) and 0 < limit < 1000000:
                limits.append(limit)
        self.max_length = min(limits)

    def predict_many(self, texts):
        if not texts:
            return []
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            lengths = [len(self.classifier.tokenizer(text, truncation=False,
                        verbose=False)["input_ids"]) for text in batch]
            predictions = self.classifier(batch, top_k=None, truncation=True,
                                          max_length=self.max_length, batch_size=self.batch_size)
            if len(predictions) != len(batch):
                raise ValueError("模型回傳筆數與輸入不一致")
            for scores, length in zip(predictions, lengths):
                probabilities = {item["label"].lower(): float(item["score"]) for item in scores}
                if (len(scores) != 3 or set(probabilities) != LABELS or
                        any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()) or
                        not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001)):
                    raise ValueError("模型必須回傳三類有效機率，且總和為 1")
                label = max(probabilities, key=probabilities.get)
                results.append({"label": label, "score": probabilities["positive"] - probabilities["negative"],
                                "confidence": probabilities[label], "probabilities": probabilities,
                                "model": self.model_name, "max_length": self.max_length,
                                "input_tokens": length,
                                "status": "truncated" if length > self.max_length else "ok"})
        return results
