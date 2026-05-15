# Adaptive Persona Engine
### AI/ML Engineer (Intern) Assessment — L2

> **All 4 parts implemented. 33/33 tests passing. Production-ready.**

---

## Architecture Overview

```
adaptive-persona-engine/
├── src/
│   ├── persona_engine/
│   │   └── drift_detector.py       # Part 1: Persona drift detection
│   ├── intent_classifier/
│   │   ├── classifier.py           # Part 2: Offline intent classifier
│   │   └── training_data.py        # Labeled training dataset
│   ├── rag_resolver/
│   │   └── conflict_resolver.py    # Part 3: RAG conflict resolver
│   └── api/
│       └── server.py               # FastAPI production server
├── data/
│   └── persona_data.json           # Sample persona JSON (Round 1 format)
├── models/
│   └── intent_classifier.joblib    # Trained model (19.6KB, <50MB ✅)
├── tests/
│   └── test_all.py                 # 33 tests covering all parts
├── docs/
│   ├── SYSTEM_DESIGN.md            # Part 4: Sync architecture design
│   └── SELF_EVALUATION.md          # Honest self-assessment
├── scripts/
│   └── prestart.py                 # Model training hook for cold starts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option A — Local (Python)

```bash
# 1. Clone
git clone https://github.com/your-username/adaptive-persona-engine
cd adaptive-persona-engine

# 2. Install
pip install -r requirements.txt

# 3. Train model + start server
python scripts/prestart.py
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 4. Visit docs
open http://localhost:8000/docs
```

### Option B — Docker

```bash
docker-compose up --build
# Server available at http://localhost:8000
```

### Run Tests

```bash
python -m pytest tests/ -v
# Expected: 33 passed
```

---

## Part 1 — Persona Drift Detector

Tracks how a user's mood and tone changes **per day** (not overall), identifies the trigger that caused each shift.

```bash
# CLI demo
python src/persona_engine/drift_detector.py data/persona_data.json

# API
curl http://localhost:8000/persona/demo
```

**Output:**
```
Day  1 (2024-01-01)  🔍 CURIOUS & 🎩 FORMAL   | Sentiment: +0.90
Day  2 (2024-01-02)  😊 HAPPY   & 🎩 FORMAL   | Sentiment: +0.49
Day  3 (2024-01-03)  😤 FRUSTRATED & 🎩 FORMAL | Sentiment: -0.61
Day  4 (2024-01-04)  😰 ANXIOUS & 👕 CASUAL   | Sentiment: -0.25
Day  7 (2024-01-07)  🎉 PLAYFUL & 🧥 NEUTRAL  | Sentiment: +0.96

DRIFT EVENTS:
  Day 2→3: [EVENT] 'rag' — Sharp sentiment shift (+0.49 → -0.61)
  Day 3→4: [PERSON] 'manager' — New entity correlated with tone shift
  Day 4→7: [EVENT] 'humor' — Sentiment recovery (+0.96)
```

**How it works:**
- `FormalityAnalyzer`: regex-based formal/casual pattern matching
- `MoodAnalyzer`: VADER sentiment + keyword scoring for 6 mood categories
- `DriftTriggerDetector`: priority cascade — person → sentiment event → topic
- `compute_drift_magnitude()`: weighted formula (40% sentiment + 30% formality + 30% mood change)

---

## Part 2 — Offline Intent Classifier

Classifies messages into 5 intents. **No API calls. CPU only. <200ms SLA.**

```bash
# CLI demo
python src/intent_classifier/classifier.py

# API
curl -X POST http://localhost:8000/intent/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Remind me to call mom at 7pm"}'
```

**Response:**
```json
{
  "intent": "reminder",
  "confidence": 0.91,
  "latency_ms": 0.7,
  "all_scores": { "reminder": 0.91, "action_item": 0.05, ... }
}
```

**Model specs:**
| Metric | Value |
|--------|-------|
| Algorithm | TF-IDF (1-3 gram) + SGDClassifier |
| Model size | **19.6 KB** (limit: 50MB) |
| Inference p50 | **0.51ms** |
| Inference p99 | **0.72ms** (limit: 200ms) |
| Labels | reminder / emotional_support / action_item / small_talk / unknown |

---

## Part 3 — RAG Conflict Resolver

Handles the hard retrieval problem: same entity appears in contradictory contexts.

```bash
# CLI demo
python src/rag_resolver/conflict_resolver.py

# API  
curl -X POST http://localhost:8000/rag/resolve \
  -H "Content-Type: application/json" \
  -d '{"query": "Did I mention anything about my sister?"}'
```

**Scoring formula:**
```
final_score = 0.35 × recency_score
            + 0.30 × emotional_intensity
            + 0.35 × relevance_score

recency_score  = exp(-0.15 × days_ago)
relevance      = BM25-style term frequency with entity boost
emotional      = VADER compound + keyword intensity
```

**Contradiction detection:**
- **Sentiment contradiction**: |Δsentiment| > 0.6 between relevant chunks
- **Temporal contradiction**: same entity in different topic contexts with sentiment shift > 0.3

---

## Part 4 — System Design

See [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) for the full 1-page design covering:
- On-device storage (SQLite + SQLCipher)
- What syncs vs what stays local (raw text NEVER syncs)
- 4-case conflict resolution with decision tree
- Sync protocol with retry backoff
- Security + privacy model

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API docs (Swagger) |
| POST | `/persona/analyze` | Analyze persona drift from JSON |
| GET | `/persona/demo` | Demo with built-in data |
| POST | `/intent/classify` | Classify single message |
| POST | `/intent/batch` | Classify up to 50 messages |
| GET | `/intent/benchmark` | Latency benchmark |
| POST | `/intent/retrain` | Retrain classifier |
| POST | `/rag/resolve` | Resolve RAG conflicts |
| GET | `/rag/demo` | Demo sister query |

---

## Design Decisions

**Why TF-IDF + SGD instead of a transformer?**
The task is short-text intent classification with 5 fixed labels. A BERT model is 400MB and takes 80ms. TF-IDF + SGD is **19.6KB and 0.7ms** — 3 orders of magnitude faster, stays well within both constraints. The right tool for the job.

**Why VADER for sentiment?**
Deterministic, no network calls, runs in microseconds, covers slang and emoticons. For a production system with labeled feedback, I'd supplement with a fine-tuned lightweight model.

**Why BM25-style scoring instead of embeddings for RAG?**
Embeddings require a model (~80MB) which would violate the spirit of the offline-first design. BM25 is explainable, fast, and effective for exact entity matching. The code architecture accepts a `chunks` parameter for hot-swapping to vector retrieval.

---

## Self-Evaluation

See [`docs/SELF_EVALUATION.md`](docs/SELF_EVALUATION.md) for an honest breakdown of what's strong, what trade-offs were made, and what would be improved with more time.

---

*Built in < 24 hours | All tests green | Production deployment ready*
