"""
Adaptive Persona Engine — Production FastAPI Server
Exposes all 3 subsystems via REST endpoints.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Annotated

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.persona_engine.drift_detector import PersonaDriftDetector, PersonaTimeline
from src.intent_classifier.classifier import IntentClassifier
from src.rag_resolver.conflict_resolver import RAGConflictResolver, PERSONA_CHUNKS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("adaptive-persona-engine")

# ---------------------------------------------------------------------------
# App Bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Persona Engine",
    description="AI/ML Intern Assessment — Parts 1, 2, 3",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_drift_detector: Optional[PersonaDriftDetector] = None
_intent_classifier: Optional[IntentClassifier] = None
_rag_resolver: Optional[RAGConflictResolver] = None


def get_drift_detector() -> PersonaDriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = PersonaDriftDetector()
    return _drift_detector


def get_intent_classifier() -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def get_rag_resolver() -> RAGConflictResolver:
    global _rag_resolver
    if _rag_resolver is None:
        _rag_resolver = RAGConflictResolver()
    return _rag_resolver


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class PersonaRequest(BaseModel):
    persona_json: Dict[str, Any] = Field(..., description="Persona JSON from Round 1")


class IntentRequest(BaseModel):
    # Pydantic v2: use Field(min_length=..., max_length=...) on the field itself
    text: str = Field(..., min_length=1, max_length=2000)


class IntentBatchRequest(BaseModel):
    # Pydantic v2: use Field(min_length=..., max_length=...) for list length constraints
    texts: List[str] = Field(..., min_length=1, max_length=50)


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


# ---------------------------------------------------------------------------
# Health & Meta
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health() -> Dict[str, Any]:
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}


@app.get("/", tags=["Meta"])
def root() -> Dict[str, Any]:
    return {
        "service": "Adaptive Persona Engine",
        "endpoints": {
            "Part 1 — Persona Drift":   "POST /persona/analyze",
            "Part 2 — Intent Classify": "POST /intent/classify",
            "Part 2 — Batch Classify":  "POST /intent/batch",
            "Part 2 — Benchmark":       "GET  /intent/benchmark",
            "Part 3 — RAG Resolve":     "POST /rag/resolve",
            "Docs":                     "GET  /docs",
        },
    }


# ---------------------------------------------------------------------------
# Part 1 — Persona Drift
# ---------------------------------------------------------------------------

@app.post("/persona/analyze", tags=["Part 1 — Persona Drift"])
def analyze_persona(request: PersonaRequest) -> Dict[str, Any]:
    """
    Analyze persona drift across days.
    Returns timeline of mood/tone changes + detected triggers.
    """
    try:
        detector = get_drift_detector()
        timeline = detector.build_timeline(request.persona_json)

        return {
            "user_id": timeline.user_id,
            "summary": timeline.summary,
            "snapshots": [
                {
                    "day":               s.day,
                    "date":              s.date,
                    "mood":              s.mood,
                    "tone":              s.tone,
                    "energy":            s.energy,
                    "sentiment_score":   s.sentiment_score,
                    "formality_score":   s.formality_score,
                    "dominant_topics":   s.dominant_topics,
                    "mentioned_entities": s.mentioned_entities,
                    "sample_text":       s.sample_text,
                }
                for s in timeline.snapshots
            ],
            "drift_events": [
                {
                    "from_day":        d.from_day,
                    "to_day":          d.to_day,
                    "drift_magnitude": d.drift_magnitude,
                    "mood_change":     list(d.mood_change),
                    "tone_change":     list(d.tone_change),
                    "trigger_type":    d.trigger_type,
                    "trigger_value":   d.trigger_value,
                    "explanation":     d.explanation,
                }
                for d in timeline.drift_events
            ],
        }
    except Exception as e:
        logger.exception("Persona analysis failed")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/persona/demo", tags=["Part 1 — Persona Drift"])
def analyze_persona_demo() -> Dict[str, Any]:
    """Analyze built-in demo persona data."""
    demo_data = {
        "user_id": "demo_user",
        "sessions": [
            {"day": 1, "date": "2024-01-01", "messages": [{"role": "user", "text": "Could you please explain this formally? I would like to understand the architecture in detail."}], "topics": ["architecture"], "entities": []},
            {"day": 4, "date": "2024-01-04", "messages": [{"role": "user", "text": "ugh this is so frustrating, i am stuck and stressed about the deadline"}], "topics": ["deadline"], "entities": []},
            {"day": 7, "date": "2024-01-07", "messages": [{"role": "user", "text": "haha this is so fun! lol finally cracked it, game over!"}], "topics": ["victory"], "entities": []}
        ]
    }
    req = PersonaRequest(persona_json=demo_data)
    return analyze_persona(req)


# ---------------------------------------------------------------------------
# Part 2 — Intent Classifier
# ---------------------------------------------------------------------------

@app.post("/intent/classify", tags=["Part 2 — Intent Classifier"])
def classify_intent(request: IntentRequest) -> Dict[str, Any]:
    """
    Classify a single message into:
    reminder | emotional_support | action_item | small_talk | unknown
    Runs fully offline, CPU-only, < 200ms guaranteed.
    """
    try:
        clf = get_intent_classifier()
        result = clf.classify(request.text)
        return result  # type: ignore[return-value]
    except Exception as e:
        logger.exception("Intent classification failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intent/batch", tags=["Part 2 — Intent Classifier"])
def classify_intent_batch(request: IntentBatchRequest) -> Dict[str, Any]:
    """Classify up to 50 messages in a single call."""
    try:
        clf = get_intent_classifier()
        results = clf.classify_batch(request.texts)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.exception("Batch classification failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intent/benchmark", tags=["Part 2 — Intent Classifier"])
def benchmark_classifier(
    n_runs: int = Query(default=100, ge=10, le=1000),
) -> Dict[str, Any]:
    """Run latency benchmark. Verifies <200ms SLA at p99."""
    try:
        clf = get_intent_classifier()
        result = clf.benchmark(n_runs)
        return result  # type: ignore[return-value]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intent/retrain", tags=["Part 2 — Intent Classifier"])
def retrain_classifier() -> Dict[str, Any]:
    """Trigger a model retrain from updated training data."""
    try:
        clf = get_intent_classifier()
        clf.retrain()
        return {"status": "retrained", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Part 3 — RAG Conflict Resolver
# ---------------------------------------------------------------------------

@app.post("/rag/resolve", tags=["Part 3 — RAG Conflict Resolver"])
def resolve_rag(request: RAGRequest) -> Dict[str, Any]:
    """
    Resolve contradictory RAG context.
    Ranks chunks by recency + emotional weight, flags contradictions,
    returns a merged coherent answer.
    """
    try:
        resolver = get_rag_resolver()
        result = resolver.resolve(request.query)

        return {
            "query":               result.query,
            "confidence":          result.confidence,
            "resolution_strategy": result.resolution_strategy,
            "merged_answer":       result.merged_answer,
            "ranked_chunks": [
                {
                    "rank":     sc.rank,
                    "chunk_id": sc.chunk.chunk_id,
                    "day":      sc.chunk.day,
                    "date":     sc.chunk.date,
                    "text":     sc.chunk.text,
                    "topic":    sc.chunk.topic,
                    "scores": {
                        "final":     sc.final_score,
                        "recency":   sc.recency_score,
                        "emotional": sc.emotional_score,
                        "relevance": sc.relevance_score,
                    },
                }
                for sc in result.ranked_chunks
            ],
            "contradictions": [
                {
                    "chunk_a_id": c.chunk_a_id,
                    "chunk_b_id": c.chunk_b_id,
                    "type":       c.conflict_type,
                    "severity":   c.severity,
                    "description": c.description,
                }
                for c in result.contradictions
            ],
        }
    except Exception as e:
        logger.exception("RAG resolution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/demo", tags=["Part 3 — RAG Conflict Resolver"])
def demo_rag() -> Dict[str, Any]:
    """Run the canonical demo query: 'Did I mention anything about my sister?'"""
    req = RAGRequest(query="Did I mention anything about my sister?")
    return resolve_rag(req)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Adaptive Persona Engine...")
    # Pre-warm all models at boot so first requests are fast
    get_drift_detector()
    get_intent_classifier()
    get_rag_resolver()
    logger.info("All models loaded. Server ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )