"""
Production test suite — covers all 3 parts end-to-end.
Run with: python -m pytest tests/ -v
"""

import sys
import json
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Part 1 — Persona Drift Detector
# ---------------------------------------------------------------------------

class TestPersonaDriftDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.persona_engine.drift_detector import PersonaDriftDetector
        self.detector = PersonaDriftDetector()
        with open("data/persona_data.json") as f:
            self.persona_json = json.load(f)

    def test_timeline_builds_without_error(self):
        timeline = self.detector.build_timeline(self.persona_json)
        assert timeline is not None
        assert timeline.user_id == "user_001"

    def test_snapshot_count_matches_sessions(self):
        timeline = self.detector.build_timeline(self.persona_json)
        assert len(timeline.snapshots) == len(self.persona_json["sessions"])

    def test_snapshots_have_required_fields(self):
        timeline = self.detector.build_timeline(self.persona_json)
        for snap in timeline.snapshots:
            assert snap.day > 0
            assert snap.mood in ["curious", "frustrated", "happy", "playful", "anxious", "satisfied", "neutral"]
            assert snap.tone in ["formal", "casual", "neutral", "playful"]
            assert -1.0 <= snap.sentiment_score <= 1.0
            assert 0.0 <= snap.formality_score <= 1.0

    def test_drift_events_detected(self):
        timeline = self.detector.build_timeline(self.persona_json)
        assert len(timeline.drift_events) > 0, "Should detect at least one drift"

    def test_drift_magnitude_in_range(self):
        timeline = self.detector.build_timeline(self.persona_json)
        for drift in timeline.drift_events:
            assert 0.0 <= drift.drift_magnitude <= 1.0

    def test_drift_trigger_types_valid(self):
        timeline = self.detector.build_timeline(self.persona_json)
        valid_types = {"topic", "event", "person"}
        for drift in timeline.drift_events:
            assert drift.trigger_type in valid_types

    def test_day1_is_formal(self):
        """Day 1 messages are formal — must be detected as such."""
        timeline = self.detector.build_timeline(self.persona_json)
        day1 = next(s for s in timeline.snapshots if s.day == 1)
        assert day1.tone == "formal"

    def test_day7_is_playful(self):
        """Day 7 messages are playful — must detect playful mood."""
        timeline = self.detector.build_timeline(self.persona_json)
        day7 = next(s for s in timeline.snapshots if s.day == 7)
        assert day7.mood in ["playful", "happy", "satisfied"]

    def test_summary_contains_all_days(self):
        timeline = self.detector.build_timeline(self.persona_json)
        assert "Day 1" in timeline.summary
        assert "Day 7" in timeline.summary

    def test_json_serializable(self):
        timeline = self.detector.build_timeline(self.persona_json)
        result_json = self.detector.to_json(timeline)
        parsed = json.loads(result_json)
        assert "snapshots" in parsed
        assert "drift_events" in parsed


# ---------------------------------------------------------------------------
# Part 2 — Intent Classifier
# ---------------------------------------------------------------------------

class TestIntentClassifier:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.intent_classifier.classifier import IntentClassifier
        self.clf = IntentClassifier()

    def test_classifies_reminder(self):
        result = self.clf.classify("Remind me to call mom at 7pm")
        assert result["intent"] == "reminder"

    def test_classifies_emotional_support(self):
        result = self.clf.classify("I've been really anxious and overwhelmed lately")
        assert result["intent"] == "emotional_support"

    def test_classifies_action_item(self):
        result = self.clf.classify("Please generate a report on sales performance")
        assert result["intent"] == "action_item"

    def test_classifies_small_talk(self):
        result = self.clf.classify("Hey! How's it going today?")
        assert result["intent"] == "small_talk"

    def test_classifies_unknown(self):
        result = self.clf.classify("xkjfhsd asdlkfj???")
        assert result["intent"] == "unknown"

    def test_confidence_in_range(self):
        result = self.clf.classify("Can you remind me about the meeting?")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_latency_under_200ms(self):
        """SLA requirement: all inferences under 200ms."""
        texts = [
            "Remind me to do something",
            "I feel so sad today",
            "Generate a PDF report",
            "Hey what's up",
            "sdfjksd???",
        ]
        for text in texts:
            result = self.clf.classify(text)
            assert result["latency_ms"] < 200, f"Latency {result['latency_ms']}ms for: {text}"

    def test_all_scores_present(self):
        result = self.clf.classify("Set a reminder for tomorrow")
        assert "all_scores" in result
        expected_labels = {"reminder", "emotional_support", "action_item", "small_talk", "unknown"}
        assert expected_labels == set(result["all_scores"].keys())

    def test_batch_classify(self):
        texts = ["Remind me", "I'm sad", "Run a report", "Hey!", "asdf"]
        results = self.clf.classify_batch(texts)
        assert len(results) == 5
        assert all("intent" in r for r in results)

    def test_model_size_under_50mb(self):
        from src.intent_classifier.classifier import MODEL_PATH
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        assert size_mb < 50.0, f"Model is {size_mb:.2f}MB — exceeds 50MB limit"

    def test_benchmark_passes_sla(self):
        bench = self.clf.benchmark(50)
        assert bench["passes_200ms_sla"] is True
        assert bench["p99_ms"] < 200


# ---------------------------------------------------------------------------
# Part 3 — RAG Conflict Resolver
# ---------------------------------------------------------------------------

class TestRAGConflictResolver:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.rag_resolver.conflict_resolver import RAGConflictResolver
        self.resolver = RAGConflictResolver()

    def test_canonical_query_returns_answer(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        assert result.merged_answer
        assert len(result.merged_answer) > 50

    def test_ranked_chunks_sorted_by_score(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        scores = [sc.final_score for sc in result.ranked_chunks]
        assert scores == sorted(scores, reverse=True)

    def test_chunks_ranked_1_to_n(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        ranks = [sc.rank for sc in result.ranked_chunks]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_contradictions_detected_for_sister_query(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        assert len(result.contradictions) > 0, "Should detect sentiment contradictions"

    def test_contradiction_types_valid(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        valid_types = {"sentiment", "factual", "temporal"}
        for c in result.contradictions:
            assert c.conflict_type in valid_types

    def test_contradiction_severity_in_range(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        for c in result.contradictions:
            assert 0.0 <= c.severity <= 1.0

    def test_confidence_in_range(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        assert 0.0 <= result.confidence <= 1.0

    def test_recency_higher_for_recent_chunks(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        chunks_by_day = sorted(result.ranked_chunks, key=lambda x: x.chunk.day)
        # Day 7 chunk should have higher recency than Day 1
        day1_chunk = next((sc for sc in chunks_by_day if sc.chunk.day == 1), None)
        day7_chunk = next((sc for sc in chunks_by_day if sc.chunk.day == 7), None)
        if day1_chunk and day7_chunk:
            assert day7_chunk.recency_score >= day1_chunk.recency_score

    def test_irrelevant_chunk_ranks_last(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        # The architecture chunk (no sister mention) should rank last
        arch_chunk = next(sc for sc in result.ranked_chunks if sc.chunk.chunk_id == "c005")
        sister_chunks = [sc for sc in result.ranked_chunks if sc.chunk.chunk_id != "c005"]
        assert all(sc.final_score >= arch_chunk.final_score for sc in sister_chunks)

    def test_resolution_strategy_set(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        assert result.resolution_strategy in [
            "contradiction_aware_merge", "recency_weighted_merge", "no_match"
        ]

    def test_json_output(self):
        result = self.resolver.resolve("Did I mention anything about my sister?")
        json_out = self.resolver.to_json(result)
        parsed = json.loads(json_out)
        assert "ranked_chunks" in parsed
        assert "contradictions" in parsed
        assert "merged_answer" in parsed


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self):
        """Run all three parts in sequence — simulates real usage."""
        import json
        from src.persona_engine.drift_detector import PersonaDriftDetector
        from src.intent_classifier.classifier import IntentClassifier
        from src.rag_resolver.conflict_resolver import RAGConflictResolver

        with open("data/persona_data.json") as f:
            persona = json.load(f)

        # Part 1
        detector = PersonaDriftDetector()
        timeline = detector.build_timeline(persona)
        assert len(timeline.snapshots) > 0

        # Part 2
        clf = IntentClassifier()
        for session in persona["sessions"]:
            for msg in session["messages"]:
                if msg["role"] == "user":
                    result = clf.classify(msg["text"])
                    assert result["intent"] in [
                        "reminder", "emotional_support", "action_item", "small_talk", "unknown"
                    ]
                    assert result["latency_ms"] < 200

        # Part 3
        resolver = RAGConflictResolver()
        result = resolver.resolve("Did I mention anything about my sister?")
        assert result.merged_answer
        assert len(result.ranked_chunks) > 0
