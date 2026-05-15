"""
Offline Intent Classifier — Part 2
- TF-IDF + SGDClassifier pipeline (model < 2MB, inference < 10ms)
- Zero external API calls
- Labels: reminder | emotional_support | action_item | small_talk | unknown
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.intent_classifier.training_data import TRAINING_DATA, LABEL_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
LABELS = ["reminder", "emotional_support", "action_item", "small_talk", "unknown"]
CONFIDENCE_THRESHOLD = 0.45  # below this → fallback to "unknown"


# ---------------------------------------------------------------------------
# Feature Engineering Helpers
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """Lightweight preprocessing — lowercase + normalize."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'?!]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def augment_training_data(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Simple augmentation: add slight typos + contractions to improve robustness."""
    augmented = list(data)
    contractions = {
        "can you": "can u", "do not": "don't", "I am": "I'm",
        "I have": "I've", "please": "pls", "remind me": "remind me to",
    }
    for text, label in data[:30]:  # augment first 30 samples
        new_text = text
        for k, v in contractions.items():
            if k.lower() in new_text.lower():
                new_text = new_text.replace(k, v)
                break
        if new_text != text:
            augmented.append((new_text, label))
    return augmented


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    TF-IDF + SGD (modified_huber loss) pipeline.
    Fast, lightweight, interpretable, and highly effective for short text.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),        # unigrams + bigrams + trigrams
            max_features=8000,
            sublinear_tf=True,         # log normalization
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )),
        ("clf", SGDClassifier(
            loss="modified_huber",     # supports predict_proba
            alpha=0.001,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            tol=1e-4,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class IntentClassifierTrainer:

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def train(self, verbose: bool = True) -> Pipeline:
        data = augment_training_data(TRAINING_DATA)
        texts = [preprocess(t) for t, _ in data]
        labels = [label for _, label in data]

        # Counter is imported at module level — always available
        dist: Counter[str] = Counter(labels)

        if verbose:
            print(f"[IntentClassifier] Training on {len(texts)} samples...")
            for k, v in sorted(dist.items()):
                print(f"  {k:25s}: {v} samples")

        pipeline = build_pipeline()

        # sklearn expects array-like; List[str] is valid but we cast for type-checkers
        pipeline.fit(texts, labels)

        # Cross-validation for confidence report
        if verbose:
            min_class_count = min(dist.values()) if dist else 1
            n_splits = min(5, min_class_count)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # Cast to np.ndarray to satisfy stricter type stubs (MatrixLike)
            texts_arr = np.array(texts)
            labels_arr = np.array(labels)
            scores = cross_val_score(
                pipeline, texts_arr, labels_arr, cv=cv, scoring="f1_macro"
            )
            print(f"\n[CV] F1-Macro: {scores.mean():.3f} ± {scores.std():.3f}")

        # Save
        joblib.dump(pipeline, MODEL_PATH, compress=3)
        size_kb = MODEL_PATH.stat().st_size / 1024
        if verbose:
            print(f"[✓] Model saved → {MODEL_PATH}  ({size_kb:.1f} KB, limit: 50MB)")
            assert size_kb < 50 * 1024, "Model exceeds 50MB!"

        return pipeline


# ---------------------------------------------------------------------------
# Classifier (Inference)
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Production-ready offline intent classifier.
    Loads once, classifies in <10ms per message.
    """

    _instance: Optional["IntentClassifier"] = None

    def __init__(self) -> None:
        # Explicitly typed — never left as None after _load_or_train()
        self._pipeline: Pipeline = self._load_or_train()

    @classmethod
    def get_instance(cls) -> "IntentClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_or_train(self) -> Pipeline:
        """Always returns a fully initialised Pipeline — never None."""
        if MODEL_PATH.exists():
            pipeline: Pipeline = joblib.load(MODEL_PATH)
            return pipeline
        trainer = IntentClassifierTrainer()
        return trainer.train(verbose=True)

    def classify(self, text: str) -> Dict[str, object]:
        """
        Returns:
            {
                "intent": str,
                "confidence": float,
                "all_scores": dict,
                "latency_ms": float,
                "fallback": bool,
                "description": str,
            }
        """
        t0 = time.perf_counter()

        processed = preprocess(text)

        # _pipeline is always a Pipeline (never None) — safe attribute access
        proba: np.ndarray = self._pipeline.predict_proba([processed])[0]
        classes: np.ndarray = self._pipeline.classes_

        scores: Dict[str, float] = {
            cls: float(prob) for cls, prob in zip(classes.tolist(), proba.tolist())
        }

        # max() over a plain dict with a str key — no overload ambiguity
        best_label: str = max(scores, key=scores.__getitem__)
        confidence: float = scores[best_label]

        # Confidence gate
        fallback = confidence < CONFIDENCE_THRESHOLD
        intent = best_label if not fallback else "unknown"

        latency_ms = (time.perf_counter() - t0) * 1000

        assert latency_ms < 200, f"Latency {latency_ms:.1f}ms exceeds 200ms SLA!"

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_scores": {
                k: round(v, 4)
                for k, v in sorted(scores.items(), key=lambda x: -x[1])
            },
            "latency_ms": round(latency_ms, 2),
            "fallback": fallback,
            "description": LABEL_DESCRIPTIONS.get(intent, ""),
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        return [self.classify(t) for t in texts]

    def retrain(self) -> None:
        trainer = IntentClassifierTrainer()
        trainer.train(verbose=True)
        self._pipeline = joblib.load(MODEL_PATH)   # reload fresh

    def benchmark(self, n_runs: int = 100) -> Dict[str, object]:
        sample = "Can you remind me to check the deployment logs at 5pm?"
        latencies: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.classify(sample)
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        return {
            "p50_ms":            round(float(np.percentile(arr, 50)), 2),
            "p95_ms":            round(float(np.percentile(arr, 95)), 2),
            "p99_ms":            round(float(np.percentile(arr, 99)), 2),
            "max_ms":            round(float(arr.max()), 2),
            "n_runs":            n_runs,
            "passes_200ms_sla":  float(np.percentile(arr, 99)) < 200,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def demo() -> None:
    print("=" * 55)
    print("  OFFLINE INTENT CLASSIFIER — Demo")
    print("=" * 55)

    # Force train
    trainer = IntentClassifierTrainer()
    trainer.train(verbose=True)

    clf = IntentClassifier()

    test_cases = [
        "Remind me to call mom at 7pm",
        "I've been feeling really down lately and don't know who to talk to",
        "Please generate a summary of this week's sales report",
        "Haha what a day! Did you see that meme?",
        "Xkjfhsdf asdkjfh???",
        "Don't let me forget to submit the PR before the deadline",
        "I'm overwhelmed and anxious about tomorrow's presentation",
        "Can you translate this document to French?",
        "Good morning! Hope your week is going well",
    ]

    print("\n📊 CLASSIFICATION RESULTS:")
    print("-" * 55)
    for text in test_cases:
        result = clf.classify(text)
        flag = "⚠️ " if result["fallback"] else "✅"
        print(
            f"{flag} [{str(result['intent']).upper():18s}] "
            f"({result['confidence']:.2f}) | {result['latency_ms']:.1f}ms"
        )
        print(f"   \"{text[:60]}\"")
        print()

    print("\n⚡ LATENCY BENCHMARK (100 runs):")
    bench = clf.benchmark(100)
    for k, v in bench.items():
        print(f"  {k:25s}: {v}")

    model_size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\n📦 Model size: {model_size_kb:.1f} KB ({model_size_kb/1024:.2f} MB)")
    print(f"   Within 50MB limit: {'✅' if model_size_kb < 50*1024 else '❌'}")


if __name__ == "__main__":
    demo()