#!/usr/bin/env python
"""
Pre-start hook: ensures the intent classifier model is trained before server starts.
Runs once per cold start. Model persisted to /app/models/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent_classifier.classifier import MODEL_PATH, IntentClassifierTrainer

if not MODEL_PATH.exists():
    print("[prestart] Model not found — training now...")
    trainer = IntentClassifierTrainer()
    trainer.train(verbose=True)
    print("[prestart] Training complete.")
else:
    print(f"[prestart] Model found at {MODEL_PATH} — skipping training.")
