# Part 4: Sync Architecture — System Design Document

**Adaptive Persona Engine | On-Device + Cloud Sync**
*Version 1.0 | Written by: Engineering Lead*

---

## 1. Overview

The Adaptive Persona Engine runs across two planes:

- **On-device (edge):** Real-time inference, local memory, offline-capable
- **Cloud (server):** Durable storage, cross-device sync, model updates

The design principle is **local-first**: the app works fully offline and syncs opportunistically when connectivity is available.

---

## 2. On-Device Storage

### What lives locally

| Data Type              | Storage          | Why Local                          |
|------------------------|------------------|------------------------------------|
| Persona snapshots      | SQLite (indexed) | Fast reads, no latency             |
| Raw conversation turns | SQLite + WAL     | Privacy-first, not transmitted raw |
| Intent classifier      | Flat file (.joblib) | <50MB, CPU-only inference       |
| Emotion scores (daily) | SQLite           | Aggregated, queryable              |
| Drift timeline JSON    | Local file       | Derived, rebuildable               |
| User preferences       | SQLite           | Low-write, high-read               |
| Pending sync queue     | SQLite (FIFO)    | Survives app crashes               |

### What is NEVER sent to cloud

- Raw conversation text (verbatim messages)
- Private entity names (family, colleagues) unless opted in
- Biometric or location data

---

## 3. What Syncs vs What Stays Local

```
┌─────────────────────────────────────────────────────────┐
│                    ON-DEVICE (Edge)                      │
│                                                          │
│  ┌───────────────┐    ┌────────────────────────────┐   │
│  │  SQLite DB    │    │  Intent Classifier         │   │
│  │  - sessions   │    │  (intent_classifier.joblib) │   │
│  │  - raw turns  │    │  <50MB, CPU-only           │   │
│  │  - snapshots  │    └────────────────────────────┘   │
│  │  - queue      │                                      │
│  └───────┬───────┘                                      │
│          │ Sync Delta (WHAT SYNCS ↓)                    │
└──────────┼──────────────────────────────────────────────┘
           │
           │  SYNCS (encrypted, anonymized):
           │  ✅ Aggregated persona snapshots (mood, tone, day)
           │  ✅ Drift event metadata (type, trigger, magnitude)
           │  ✅ Intent classification labels (not raw text)
           │  ✅ Topic clusters (anonymized keywords)
           │  ✅ Model version + checksum
           │  ❌ Raw message text (NEVER)
           │  ❌ Entity names (opt-in only)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                    CLOUD (Server)                        │
│                                                          │
│  ┌────────────────┐   ┌──────────────────────────────┐ │
│  │  PostgreSQL    │   │  Object Storage (S3/GCS)     │ │
│  │  - user index  │   │  - model artifacts (.joblib)  │ │
│  │  - snapshots   │   │  - aggregate analytics       │ │
│  │  - sync log    │   └──────────────────────────────┘ │
│  └────────────────┘                                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  FastAPI Sync Endpoints                            │ │
│  │  POST /sync/push    — device → cloud               │ │
│  │  GET  /sync/pull    — cloud → device (delta)       │ │
│  │  GET  /sync/model   — model update check           │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Conflict Resolution Strategy

### Problem
User uses the app on two devices (phone + laptop) while offline. Both create drift snapshots for Day 5. When connectivity returns, which wins?

### Resolution: Last-Write-Wins + Merge Strategy

```
CONFLICT TYPES:
─────────────────────────────────────────────────────────

1. PERSONA SNAPSHOT CONFLICT (same day, different device)
   Strategy: MERGE
   - Take mean of sentiment_score, formality_score
   - Union of dominant_topics, mentioned_entities
   - Prefer higher emotional_intensity snapshot for mood label
   - Flag merged=true in record

2. DRIFT EVENT CONFLICT (same day range, different trigger)
   Strategy: APPEND + FLAG
   - Keep both events, mark one as "alternative"
   - Surface both to user in UI with confidence scores
   - Do NOT silently drop either

3. INTENT LABEL CONFLICT (same message, different label)
   Strategy: HIGHER CONFIDENCE WINS
   - Compare confidence float of each device's classification
   - Keep higher-confidence result, discard lower
   - Log discarded label for model retraining signal

4. SYNC QUEUE RACE (two pushes arrive simultaneously)
   Strategy: OPTIMISTIC LOCKING
   - Each sync payload carries device_id + last_sync_timestamp
   - Server uses IF updated_at = :expected UPDATE
   - If stale → return 409 Conflict → device re-fetches + retries
```

### Conflict Resolution Decision Tree

```
Incoming sync record
        │
        ▼
  Record exists?
    │         │
   NO        YES
    │         │
  INSERT    Compare timestamps
              │
     ┌────────┼─────────────┐
  Same      Newer         Older
   day     incoming      incoming
    │         │              │
  MERGE    OVERWRITE     DISCARD
  (avg)    (log old)    (log new for audit)
```

---

## 5. Sync Protocol

```
DEVICE SYNC HANDSHAKE:

1. Device  →  Server : GET /sync/status?device_id=X&last_sync_ts=T
2. Server  →  Device : { needs_push: bool, needs_pull: bool, model_version: "1.2" }
3. Device  →  Server : POST /sync/push { payload: [...snapshots], checksum: SHA256 }
4. Server  validates  checksum, applies conflict resolution
5. Server  →  Device : GET /sync/pull?since=T → delta of server-side changes
6. Device  applies   delta to local SQLite
7. Device  updates   last_sync_ts = now()

RETRY POLICY:
- Exponential backoff: 1s → 2s → 4s → 8s → max 60s
- Max retries: 5 per session
- Offline: queue locally, sync on next connectivity event
```

---

## 6. Security & Privacy

| Concern              | Mitigation                                        |
|----------------------|---------------------------------------------------|
| Data in transit      | TLS 1.3 enforced                                  |
| Data at rest (cloud) | AES-256 encryption                                |
| Data at rest (device)| SQLCipher (SQLite encryption)                     |
| PII leakage          | Raw text never leaves device                      |
| Model integrity      | SHA256 checksum on download                       |
| Auth                 | JWT + refresh token rotation                      |
| GDPR compliance      | DELETE /user/:id wipes all cloud data permanently |

---

## 7. Architecture Diagram (ASCII)

```
╔══════════════════════════════════════════════════════════════╗
║                  ADAPTIVE PERSONA ENGINE                     ║
║                    SYNC ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [Phone]               [Laptop]              [Cloud]         ║
║  ┌────────────┐        ┌────────────┐        ┌───────────┐  ║
║  │ SQLite     │        │ SQLite     │        │PostgreSQL │  ║
║  │ Classifier │◄──────►│ Classifier │◄──────►│ + S3      │  ║
║  │ FastAPI    │  SYNC  │ FastAPI    │  SYNC  │ FastAPI   │  ║
║  └────────────┘        └────────────┘        └───────────┘  ║
║       │                      │                     │         ║
║  [Local Inference]      [Local Inference]     [Model Hub]   ║
║  < 200ms, offline       < 200ms, offline     (versioned)    ║
║                                                              ║
║  CONFLICT RESOLUTION:                                        ║
║  Merge snapshots → Append drift events → Confidence wins     ║
║  for intents → Optimistic locking for races                  ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 8. Trade-offs

| Decision                     | Pro                              | Con                                  |
|------------------------------|----------------------------------|--------------------------------------|
| Local-first architecture     | Offline-capable, low latency     | Sync complexity, merge conflicts     |
| TF-IDF + SGD (no LLM)        | <50MB, CPU-only, <1ms inference  | Lower accuracy on edge cases         |
| VADER for sentiment          | No API, deterministic, fast      | Misses nuanced sarcasm               |
| Raw text stays local         | Strong privacy, GDPR-simple      | No server-side NLP improvement       |
| Optimistic locking for sync  | Simple, low server load          | Requires retry logic on device       |
| SQLite on device             | Zero config, ACID, fast          | Not suited for >10k sessions/day     |

---

*End of System Design Document — Part 4*
