"""
Persona Drift Detector — Part 1
Tracks mood/tone changes across days, identifies drift triggers,
outputs a structured timeline with emotional + linguistic signals.
Production-grade: no unbound variables, full type safety, flexible input.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import math

# VADER for sentiment (fully offline, no API calls)
_vader_analyzer: Optional[Any] = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderClass
    _vader_analyzer = _VaderClass()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PersonaSnapshot:
    day: int
    date: str
    tone: str                       # formal / casual / neutral / playful
    mood: str                       # curious / frustrated / happy / anxious / neutral
    energy: str                     # high / medium / low
    sentiment_score: float          # -1.0 to 1.0
    formality_score: float          # 0.0 to 1.0
    dominant_topics: List[str]
    mentioned_entities: List[str]
    sample_text: str
    raw_labels: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftEvent:
    from_day: int
    to_day: int
    drift_magnitude: float          # 0.0 to 1.0
    mood_change: Tuple[str, str]
    tone_change: Tuple[str, str]
    trigger_type: str               # topic | event | person
    trigger_value: str
    explanation: str


@dataclass
class PersonaTimeline:
    user_id: str
    snapshots: List[PersonaSnapshot]
    drift_events: List[DriftEvent]
    summary: str


# ---------------------------------------------------------------------------
# Feature Extractors
# ---------------------------------------------------------------------------

class FormalityAnalyzer:
    """Rule-based formality scorer — fully offline, no ML required."""

    FORMAL_PATTERNS = [
        r'\b(I would|I shall|I am|Could you|Please|Thank you|Furthermore|Moreover|However|Additionally|In addition|I have)\b',
        r'\b(kindly|regarding|pertaining|herewith|accordingly|subsequently)\b',
        r'[.]{1}(?!\.)(?!\d)',
    ]

    CASUAL_PATTERNS = [
        r'\b(hey|yeah|nah|gonna|wanna|kinda|sorta|lol|haha|omg|btw|tbh|ugh|okay so)\b',
        r'[!]{2,}',
        r'[?]{2,}',
        r'\.{3}',
        r'😂|😅|😭|🙈|🔥|💀',
        r'\bfr\b|\bngl\b|\bidk\b',
    ]

    def score(self, text: str) -> float:
        """Returns 0.0 (very casual) to 1.0 (very formal)."""
        formal_hits = sum(
            len(re.findall(p, text, re.IGNORECASE))
            for p in self.FORMAL_PATTERNS
        )
        casual_hits = sum(
            len(re.findall(p, text.lower()))
            for p in self.CASUAL_PATTERNS
        )
        total = formal_hits + casual_hits
        if total == 0:
            return 0.5
        return round(formal_hits / total, 3)

    def classify(self, score: float) -> str:
        if score >= 0.7:
            return "formal"
        elif score >= 0.4:
            return "neutral"
        return "casual"


class MoodAnalyzer:
    """Sentiment + keyword-based mood detector. Fully offline."""

    MOOD_KEYWORDS: Dict[str, List[str]] = {
        "frustrated": ["ugh", "frustrating", "confused", "hate", "stressed", "stuck", "annoying", "argh", "circles"],
        "curious":    ["understand", "how", "why", "explain", "learn", "interested", "wondering", "curious", "architecture"],
        "happy":      ["great", "excellent", "love", "wonderful", "amazing", "fantastic", "clicked", "cracked", "worth it"],
        "playful":    ["haha", "lol", "funny", "priceless", "game", "fun", "silly", "quiz"],
        "anxious":    ["deadline", "stressed", "worried", "pressure", "nervous", "overwhelmed", "breathing down"],
        "satisfied":  ["finally", "done", "finished", "proud", "nailed", "sorted", "feeling great"],
    }

    def sentiment_score(self, text: str) -> float:
        """Returns compound sentiment in range -1.0 to 1.0."""
        if VADER_AVAILABLE and _vader_analyzer is not None:
            scores = _vader_analyzer.polarity_scores(text)
            return round(float(scores["compound"]), 3)

        # Fallback: simple word-count heuristic
        positive = {"good", "great", "love", "excellent", "happy", "joy", "fun", "best"}
        negative = {"bad", "hate", "awful", "terrible", "frustrated", "sad", "worst", "ugh"}
        words = text.lower().split()
        pos = sum(1 for w in words if w in positive)
        neg = sum(1 for w in words if w in negative)
        if pos + neg == 0:
            return 0.0
        return round((pos - neg) / (pos + neg), 3)

    def classify_mood(self, text: str, sentiment: float) -> Tuple[str, Dict[str, float]]:
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for mood, keywords in self.MOOD_KEYWORDS.items():
            scores[mood] = float(sum(1 for kw in keywords if kw in text_lower))

        # Boost based on sentiment polarity
        if sentiment > 0.4:
            scores["happy"]     = scores.get("happy", 0.0) + 2.0
            scores["satisfied"] = scores.get("satisfied", 0.0) + 1.0
        elif sentiment < -0.3:
            scores["frustrated"] = scores.get("frustrated", 0.0) + 2.0
            scores["anxious"]    = scores.get("anxious", 0.0) + 1.0

        if not any(v > 0 for v in scores.values()):
            return "neutral", scores

        # FIX: explicit lambda avoids type-checker unbound issues
        dominant = max(scores.keys(), key=lambda k: scores[k])
        return dominant, scores


class EnergyAnalyzer:
    """Infer energy level from message length, punctuation, and pace."""

    def classify(self, messages: List[str]) -> str:
        if not messages:
            return "medium"
        avg_len = sum(len(m) for m in messages) / len(messages)
        exclamation_count = sum(m.count("!") for m in messages)
        ellipsis_count = sum(m.count("...") for m in messages)

        if avg_len > 120 and exclamation_count > 1:
            return "high"
        elif ellipsis_count > 1 or avg_len < 60:
            return "low"
        return "medium"


# ---------------------------------------------------------------------------
# Trigger Detector
# ---------------------------------------------------------------------------

class DriftTriggerDetector:
    """Identifies what caused a persona drift between two snapshots."""

    def detect(
        self,
        prev: PersonaSnapshot,
        curr: PersonaSnapshot,
        session_data: Dict,
    ) -> Tuple[str, str, str]:
        """Returns (trigger_type, trigger_value, explanation)."""

        # 1. Person trigger — new named entity appeared
        new_entities = set(curr.mentioned_entities) - set(prev.mentioned_entities)
        if new_entities:
            entity = sorted(new_entities)[0]
            return (
                "person",
                entity,
                f"Mention of '{entity}' correlated with tone shift "
                f"{prev.tone} → {curr.tone}",
            )

        # 2. Event trigger — significant sentiment jump
        sentiment_delta = curr.sentiment_score - prev.sentiment_score
        if abs(sentiment_delta) > 0.4:
            direction = "positive event" if sentiment_delta > 0 else "negative event / stress"
            event_hint = curr.dominant_topics[0] if curr.dominant_topics else "unknown"
            return (
                "event",
                event_hint,
                f"Sharp sentiment shift ({prev.sentiment_score:+.2f} → "
                f"{curr.sentiment_score:+.2f}) linked to '{event_hint}' ({direction})",
            )

        # 3. Topic trigger — topic domain changed
        new_topics = set(curr.dominant_topics) - set(prev.dominant_topics)
        if new_topics:
            topic = sorted(new_topics)[0]
            return (
                "topic",
                topic,
                f"Introduction of new topic '{topic}' shifted mood "
                f"{prev.mood} → {curr.mood}",
            )

        # Fallback
        fallback_topic = curr.dominant_topics[0] if curr.dominant_topics else "unknown"
        return (
            "topic",
            fallback_topic,
            f"Gradual tone evolution from day {prev.day} to day {curr.day}",
        )


# ---------------------------------------------------------------------------
# Drift Magnitude
# ---------------------------------------------------------------------------

def compute_drift_magnitude(prev: PersonaSnapshot, curr: PersonaSnapshot) -> float:
    """
    Drift score = weighted combination:
      40% sentiment delta  +  30% formality delta  +  30% mood category change
    """
    sentiment_delta = abs(curr.sentiment_score - prev.sentiment_score) / 2.0
    formality_delta = abs(curr.formality_score - prev.formality_score)
    mood_changed = 0.0 if prev.mood == curr.mood else 1.0

    magnitude = (0.4 * sentiment_delta) + (0.3 * formality_delta) + (0.3 * mood_changed)
    return round(min(magnitude, 1.0), 3)


# ---------------------------------------------------------------------------
# Input Normaliser  ← fixes KeyError: 'sessions'
# ---------------------------------------------------------------------------

def _normalise_input(persona_json: Dict) -> Tuple[str, List[Dict]]:
    """
    Accepts multiple schema shapes:
      { "user_id": "...", "sessions": [...] }           ← original
      { "user_id": "...", "days": [...] }               ← alternate
      { "user_id": "...", "data": [...] }               ← alternate
      { "user_id": "...", "messages": [...] }           ← flat list
      A list of session dicts directly                  ← raw list
    Each session/day dict is also normalised to ensure
    it has 'day', 'date', 'messages', 'topics', 'entities'.
    """
    if isinstance(persona_json, list):
        sessions_raw = persona_json
        user_id = "unknown"
    else:
        user_id = persona_json.get("user_id", "unknown")
        sessions_raw = (
            persona_json.get("sessions")
            or persona_json.get("days")
            or persona_json.get("data")
            or persona_json.get("messages")
            or []
        )

    sessions: List[Dict] = []
    for idx, s in enumerate(sessions_raw):
        # Normalise message format
        raw_msgs = s.get("messages", [])
        normalised_msgs = []
        for m in raw_msgs:
            if isinstance(m, str):
                normalised_msgs.append({"role": "user", "text": m})
            elif isinstance(m, dict):
                # Ensure 'text' key exists
                if "text" not in m:
                    m["text"] = m.get("content", m.get("message", ""))
                normalised_msgs.append(m)

        sessions.append({
            "day":      s.get("day", idx + 1),
            "date":     s.get("date", s.get("timestamp", f"Day-{idx + 1}")),
            "messages": normalised_msgs,
            "topics":   s.get("topics", s.get("dominant_topics", [])),
            "entities": s.get("entities", s.get("mentioned_entities", [])),
        })

    return user_id, sessions


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class PersonaDriftDetector:

    DRIFT_THRESHOLD = 0.25   # minimum magnitude to flag as a drift event

    def __init__(self) -> None:
        self.formality        = FormalityAnalyzer()
        self.mood_analyzer    = MoodAnalyzer()
        self.energy_analyzer  = EnergyAnalyzer()
        self.trigger_detector = DriftTriggerDetector()

    # ------------------------------------------------------------------
    def analyze_session(self, session: Dict) -> PersonaSnapshot:
        user_messages = [
            m["text"]
            for m in session.get("messages", [])
            if m.get("role", "user") == "user" and m.get("text", "").strip()
        ]

        if not user_messages:
            # Session has no user messages — return neutral baseline
            return PersonaSnapshot(
                day=session.get("day", 0),
                date=session.get("date", "unknown"),
                tone="neutral", mood="neutral", energy="low",
                sentiment_score=0.0, formality_score=0.5,
                dominant_topics=session.get("topics", []),
                mentioned_entities=session.get("entities", []),
                sample_text="", raw_labels={},
            )

        full_text = " ".join(user_messages)
        sentiment      = self.mood_analyzer.sentiment_score(full_text)
        mood, mood_scores = self.mood_analyzer.classify_mood(full_text, sentiment)
        formality_score = sum(
            self.formality.score(m) for m in user_messages
        ) / len(user_messages)
        tone   = self.formality.classify(formality_score)
        energy = self.energy_analyzer.classify(user_messages)

        return PersonaSnapshot(
            day=session["day"],
            date=session["date"],
            tone=tone,
            mood=mood,
            energy=energy,
            sentiment_score=sentiment,
            formality_score=round(formality_score, 3),
            dominant_topics=session.get("topics", []),
            mentioned_entities=session.get("entities", []),
            sample_text=user_messages[0][:120],
            raw_labels=mood_scores,
        )

    # ------------------------------------------------------------------
    def detect_drifts(
        self,
        snapshots: List[PersonaSnapshot],
        sessions: List[Dict],
    ) -> List[DriftEvent]:
        drifts: List[DriftEvent] = []
        session_map = {s["day"]: s for s in sessions}

        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            magnitude = compute_drift_magnitude(prev, curr)

            if magnitude >= self.DRIFT_THRESHOLD:
                trigger_type, trigger_value, explanation = self.trigger_detector.detect(
                    prev, curr, session_map.get(curr.day, {})
                )
                drifts.append(DriftEvent(
                    from_day=prev.day,
                    to_day=curr.day,
                    drift_magnitude=magnitude,
                    mood_change=(prev.mood, curr.mood),
                    tone_change=(prev.tone, curr.tone),
                    trigger_type=trigger_type,
                    trigger_value=trigger_value,
                    explanation=explanation,
                ))

        return drifts

    # ------------------------------------------------------------------
    def build_timeline(self, persona_json: Dict) -> PersonaTimeline:
        """
        Entry point — accepts any supported schema shape.
        Raises ValueError with a clear message if input is empty.
        """
        user_id, sessions = _normalise_input(persona_json)

        if not sessions:
            raise ValueError(
                "No sessions found in persona_json. "
                "Expected a 'sessions', 'days', 'data', or 'messages' list."
            )

        snapshots = [self.analyze_session(s) for s in sessions]
        drifts    = self.detect_drifts(snapshots, sessions)

        summary = " | ".join(
            f"Day {s.day} → {s.mood} & {s.tone}" for s in snapshots
        )

        return PersonaTimeline(
            user_id=user_id,
            snapshots=snapshots,
            drift_events=drifts,
            summary=summary,
        )

    # ------------------------------------------------------------------
    def render_timeline(self, timeline: PersonaTimeline) -> str:
        TONE_EMOJI  = {"formal": "🎩", "casual": "👕", "neutral": "🧥", "playful": "🎭"}
        MOOD_EMOJI  = {
            "curious": "🔍", "frustrated": "😤", "happy": "😊",
            "playful": "🎉", "anxious": "😰", "satisfied": "✅", "neutral": "😐",
        }

        lines = [
            "=" * 65,
            f"  PERSONA DRIFT TIMELINE — User: {timeline.user_id}",
            "=" * 65,
            "",
            "📅 DAY-BY-DAY SNAPSHOT:",
        ]

        for snap in timeline.snapshots:
            t_emoji = TONE_EMOJI.get(snap.tone, "📌")
            m_emoji = MOOD_EMOJI.get(snap.mood, "💬")
            lines.append(
                f"  Day {snap.day:2d} ({snap.date})  "
                f"{m_emoji} {snap.mood.upper()} & {t_emoji} {snap.tone.upper()}"
                f"  |  Sentiment: {snap.sentiment_score:+.2f}  |  Energy: {snap.energy}"
            )

        lines += ["", "⚡ DETECTED DRIFT EVENTS:"]

        if not timeline.drift_events:
            lines.append("  No significant drifts detected.")
        else:
            for evt in timeline.drift_events:
                lines += [
                    "",
                    f"  Day {evt.from_day} → Day {evt.to_day}  [Magnitude: {evt.drift_magnitude:.2f}]",
                    f"    Mood:    {evt.mood_change[0]} → {evt.mood_change[1]}",
                    f"    Tone:    {evt.tone_change[0]} → {evt.tone_change[1]}",
                    f"    Trigger: [{evt.trigger_type.upper()}] '{evt.trigger_value}'",
                    f"    Reason:  {evt.explanation}",
                ]

        lines += ["", f"📋 SUMMARY: {timeline.summary}", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_json(self, timeline: PersonaTimeline) -> str:
        return json.dumps({
            "user_id":      timeline.user_id,
            "summary":      timeline.summary,
            "snapshots":    [asdict(s) for s in timeline.snapshots],
            "drift_events": [asdict(d) for d in timeline.drift_events],
        }, indent=2)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def run(persona_path: str = "data/persona_data.json") -> PersonaTimeline:
    with open(persona_path, encoding="utf-8") as f:
        persona_json = json.load(f)

    engine   = PersonaDriftDetector()
    timeline = engine.build_timeline(persona_json)

    print(engine.render_timeline(timeline))

    out_path = persona_path.replace(".json", "_timeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(engine.to_json(timeline))
    print(f"[✓] Timeline saved → {out_path}")

    return timeline


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/persona_data.json"
    run(path)