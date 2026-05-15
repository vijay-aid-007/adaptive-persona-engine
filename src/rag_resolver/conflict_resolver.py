"""
RAG Conflict Resolver — Part 3
Handles contradictory context across topic checkpoints.
Ranks chunks by recency + emotional weight, flags contradictions,
returns a merged coherent answer.
"""

import json
import re
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A retrieved memory chunk from the RAG store."""
    chunk_id: str
    day: int
    date: str
    text: str
    topic: str
    entities: List[str]
    sentiment_score: float      # -1.0 to 1.0
    emotional_intensity: float  # 0.0 to 1.0 (abs value + keyword boost)
    source_message_id: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    recency_score: float
    emotional_score: float
    relevance_score: float
    final_score: float
    rank: int


@dataclass
class Contradiction:
    chunk_a_id: str
    chunk_b_id: str
    conflict_type: str          # sentiment | factual | temporal
    description: str
    severity: float             # 0.0 to 1.0


@dataclass
class ResolvedAnswer:
    query: str
    ranked_chunks: List[ScoredChunk]
    contradictions: List[Contradiction]
    merged_answer: str
    confidence: float
    resolution_strategy: str


# ---------------------------------------------------------------------------
# Simulated RAG Store
# ---------------------------------------------------------------------------

PERSONA_CHUNKS: List[Chunk] = [
    Chunk(
        chunk_id="c001",
        day=2,
        date="2024-01-02",
        text="My sister called last night and asked about the same thing! Funny coincidence. We had a great chat.",
        topic="family",
        entities=["sister"],
        sentiment_score=0.72,
        emotional_intensity=0.6,
        source_message_id="m5",
    ),
    Chunk(
        chunk_id="c002",
        day=3,
        date="2024-01-03",
        text="My sister recommended a book on vector databases but I can't find it anywhere. Getting frustrated.",
        topic="frustration",
        entities=["sister"],
        sentiment_score=-0.35,
        emotional_intensity=0.7,
        source_message_id="m8",
    ),
    Chunk(
        chunk_id="c003",
        day=4,
        date="2024-01-04",
        text="btw did I mention my sister is coming to visit next week? she wants to see my project",
        topic="family",
        entities=["sister"],
        sentiment_score=0.51,
        emotional_intensity=0.5,
        source_message_id="m11",
    ),
    Chunk(
        chunk_id="c004",
        day=7,
        date="2024-01-07",
        text="Haha I just tried explaining embeddings to my sister and her face was priceless! She had no idea.",
        topic="humor",
        entities=["sister"],
        sentiment_score=0.88,
        emotional_intensity=0.85,
        source_message_id="m13",
    ),
    # Unrelated chunk — should rank low for "sister" query
    Chunk(
        chunk_id="c005",
        day=1,
        date="2024-01-01",
        text="I would like to understand how this memory system works. Could you explain the architecture?",
        topic="architecture",
        entities=[],
        sentiment_score=0.10,
        emotional_intensity=0.1,
        source_message_id="m1",
    ),
]


# ---------------------------------------------------------------------------
# Relevance Scorer
# ---------------------------------------------------------------------------

class RelevanceScorer:
    """BM25-inspired keyword relevance without external deps."""

    def score(self, query: str, chunk_text: str) -> float:
        query_terms = re.findall(r'\w+', query.lower())
        text_words = re.findall(r'\w+', chunk_text.lower())
        text_len = len(text_words)

        # TF-IDF-like term frequency with length normalization
        k1, b, avg_dl = 1.5, 0.75, 20.0
        score = 0.0
        for term in query_terms:
            tf = text_words.count(term)
            if tf == 0:
                continue
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * text_len / avg_dl))
            score += tf_norm

        # Boost for entity match
        query_entities = re.findall(
            r'\b(sister|brother|mom|dad|friend|manager|boss|colleague)\b',
            query.lower(),
        )
        for ent in query_entities:
            if ent in chunk_text.lower():
                score += 2.5

        return round(score, 4)


# ---------------------------------------------------------------------------
# Recency Scorer
# ---------------------------------------------------------------------------

class RecencyScorer:
    """Exponential decay — recent chunks score higher."""

    def __init__(self, decay_rate: float = 0.15) -> None:
        self.decay_rate = decay_rate

    def score(self, chunk: Chunk, max_day: int) -> float:
        days_ago = max(0, max_day - chunk.day)
        return round(math.exp(-self.decay_rate * days_ago), 4)


# ---------------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------------

class ContradictionDetector:

    SENTIMENT_FLIP_THRESHOLD = 0.6  # sentiment delta to flag as contradiction

    def detect(self, chunks: List[ScoredChunk]) -> List[Contradiction]:
        contradictions: List[Contradiction] = []
        relevant = [sc for sc in chunks if sc.relevance_score > 0.5]

        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                a, b = relevant[i].chunk, relevant[j].chunk

                # Sentiment contradiction
                delta = abs(a.sentiment_score - b.sentiment_score)
                if delta >= self.SENTIMENT_FLIP_THRESHOLD:
                    contradictions.append(Contradiction(
                        chunk_a_id=a.chunk_id,
                        chunk_b_id=b.chunk_id,
                        conflict_type="sentiment",
                        description=(
                            f"Day {a.day}: '{a.text[:60]}...' (sentiment: {a.sentiment_score:+.2f}) "
                            f"contradicts Day {b.day}: '{b.text[:60]}...' (sentiment: {b.sentiment_score:+.2f})"
                        ),
                        severity=round(min(delta / 2.0, 1.0), 3),
                    ))

                # Temporal contradiction — same entity, different context
                if set(a.entities) & set(b.entities) and a.topic != b.topic:
                    if abs(a.sentiment_score - b.sentiment_score) > 0.3:
                        contradictions.append(Contradiction(
                            chunk_a_id=a.chunk_id,
                            chunk_b_id=b.chunk_id,
                            conflict_type="temporal",
                            description=(
                                f"Same entity appears in conflicting contexts: "
                                f"Day {a.day} ({a.topic}) vs Day {b.day} ({b.topic})"
                            ),
                            severity=round(abs(a.sentiment_score - b.sentiment_score) / 2.0, 3),
                        ))

        # Deduplicate by chunk pair + conflict type
        seen: set[tuple[str, ...]] = set()
        unique: List[Contradiction] = []
        for c in contradictions:
            key = tuple(sorted([c.chunk_a_id, c.chunk_b_id]) + [c.conflict_type])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique


# ---------------------------------------------------------------------------
# Answer Merger
# ---------------------------------------------------------------------------

class AnswerMerger:
    """Synthesizes ranked, potentially contradictory chunks into a coherent answer."""

    def merge(
        self,
        query: str,
        ranked: List[ScoredChunk],
        contradictions: List[Contradiction],
    ) -> Tuple[str, str]:
        """Returns (merged_answer, resolution_strategy)."""

        top_chunks = [sc for sc in ranked[:4] if sc.relevance_score > 0]
        if not top_chunks:
            return "No relevant mentions found.", "no_match"

        has_contradictions = len(contradictions) > 0

        if has_contradictions:
            strategy = "contradiction_aware_merge"
            intro = (
                "Across your conversations, you mentioned your sister in different "
                "contexts — here's a chronological picture:\n"
            )
        else:
            strategy = "recency_weighted_merge"
            intro = "Here's what you mentioned about your sister, from earliest to most recent:\n"

        timeline_parts: List[str] = []
        for sc in sorted(top_chunks, key=lambda x: x.chunk.day):
            c = sc.chunk
            sentiment_label = (
                "positively" if c.sentiment_score > 0.3
                else "negatively" if c.sentiment_score < -0.2
                else "neutrally"
            )
            timeline_parts.append(
                f"• Day {c.day} ({c.date}): {c.text.strip()} "
                f"[{sentiment_label}, topic: {c.topic}]"
            )

        body = "\n".join(timeline_parts)

        if has_contradictions:
            conflict_note = f"\n\n⚠️ CONTRADICTIONS DETECTED ({len(contradictions)}):\n"
            for cont in contradictions[:2]:  # show top 2
                conflict_note += f"  • {cont.conflict_type.upper()}: {cont.description[:120]}...\n"
            most_recent_day = sorted(top_chunks, key=lambda x: -x.chunk.day)[0].chunk.day
            conflict_note += (
                "\nResolution: The most recent and emotionally prominent mention "
                f"(Day {most_recent_day}) is weighted highest."
            )
        else:
            conflict_note = ""

        # Final synthesis
        most_recent = sorted(top_chunks, key=lambda x: -x.chunk.day)[0]
        synthesis = (
            f"\nMost recent context (Day {most_recent.chunk.day}): "
            f'"{most_recent.chunk.text.strip()}"'
        )

        merged = intro + body + conflict_note + synthesis
        return merged, strategy


# ---------------------------------------------------------------------------
# Main Resolver
# ---------------------------------------------------------------------------

class RAGConflictResolver:

    WEIGHTS: Dict[str, float] = {
        "recency":   0.35,
        "emotional": 0.30,
        "relevance": 0.35,
    }

    def __init__(self, chunks: Optional[List[Chunk]] = None) -> None:
        # Fix: use the default store when chunks is None — never store None
        self.chunks: List[Chunk] = chunks if chunks is not None else PERSONA_CHUNKS
        self.relevance_scorer      = RelevanceScorer()
        self.recency_scorer        = RecencyScorer()
        self.contradiction_detector = ContradictionDetector()
        self.answer_merger         = AnswerMerger()

    def resolve(self, query: str) -> ResolvedAnswer:
        max_day = max(c.day for c in self.chunks)

        # Step 1: Score all chunks
        scored: List[ScoredChunk] = []
        for chunk in self.chunks:
            rec = self.recency_scorer.score(chunk, max_day)
            emo = chunk.emotional_intensity
            rel = self.relevance_scorer.score(query, chunk.text)

            # Normalize relevance to 0-1
            rel_norm = min(rel / 5.0, 1.0)

            final = (
                self.WEIGHTS["recency"]   * rec
                + self.WEIGHTS["emotional"] * emo
                + self.WEIGHTS["relevance"] * rel_norm
            )

            scored.append(ScoredChunk(
                chunk=chunk,
                recency_score=rec,
                emotional_score=emo,
                relevance_score=rel_norm,
                final_score=round(final, 4),
                rank=0,
            ))

        # Step 2: Rank
        scored.sort(key=lambda x: -x.final_score)
        for i, sc in enumerate(scored):
            sc.rank = i + 1

        # Step 3: Detect contradictions in top-K
        contradictions = self.contradiction_detector.detect(scored[:5])

        # Step 4: Merge answer
        merged_answer, strategy = self.answer_merger.merge(query, scored, contradictions)

        # Confidence: average final score of top relevant chunks
        top_relevant = [sc for sc in scored[:4] if sc.relevance_score > 0.1]
        confidence = round(
            sum(sc.final_score for sc in top_relevant) / max(len(top_relevant), 1), 3
        )

        return ResolvedAnswer(
            query=query,
            ranked_chunks=scored,
            contradictions=contradictions,
            merged_answer=merged_answer,
            confidence=confidence,
            resolution_strategy=strategy,
        )

    def render(self, result: ResolvedAnswer) -> str:
        lines = [
            "=" * 65,
            "  RAG CONFLICT RESOLVER",
            "=" * 65,
            f'\n🔍 QUERY: "{result.query}"',
            f"📊 Strategy: {result.resolution_strategy}  |  Confidence: {result.confidence:.3f}",
            "",
            "📋 RANKED CHUNKS:",
        ]

        for sc in result.ranked_chunks:
            relevance_bar = "█" * int(sc.relevance_score * 10)
            lines.append(
                f"  #{sc.rank} [Day {sc.chunk.day}]  "
                f"Final:{sc.final_score:.3f}  "
                f"Rec:{sc.recency_score:.2f}  "
                f"Emo:{sc.emotional_score:.2f}  "
                f"Rel:{sc.relevance_score:.2f}  "
                f"|{relevance_bar:<10}|"
            )
            lines.append(f'       "{sc.chunk.text[:80]}..."')

        if result.contradictions:
            lines += ["", f"⚠️  CONTRADICTIONS FLAGGED: {len(result.contradictions)}"]
            for c in result.contradictions:
                lines.append(
                    f"  [{c.conflict_type.upper()}] Severity: {c.severity:.2f} — {c.description[:100]}"
                )

        lines += ["", "✅ MERGED ANSWER:", "-" * 55, result.merged_answer, ""]
        return "\n".join(lines)

    def to_json(self, result: ResolvedAnswer) -> str:
        return json.dumps(
            {
                "query":               result.query,
                "confidence":          result.confidence,
                "resolution_strategy": result.resolution_strategy,
                "ranked_chunks": [
                    {
                        "rank":        sc.rank,
                        "chunk_id":    sc.chunk.chunk_id,
                        "day":         sc.chunk.day,
                        "text":        sc.chunk.text,
                        "final_score": sc.final_score,
                    }
                    for sc in result.ranked_chunks
                ],
                "contradictions": [asdict(c) for c in result.contradictions],
                "merged_answer":  result.merged_answer,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def demo() -> None:
    resolver = RAGConflictResolver()

    queries = [
        "Did I mention anything about my sister?",
        "What did I say about my sister and the book?",
        "How do I feel about my sister visiting?",
    ]

    for query in queries:
        result = resolver.resolve(query)
        print(resolver.render(result))
        print("\n")


if __name__ == "__main__":
    demo()