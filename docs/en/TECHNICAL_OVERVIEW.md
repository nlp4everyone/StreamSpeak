# Streaming Vietnamese Speech-to-Text

Production-ready multi-user streaming Speech-to-Text architecture using:

- Silero VAD (CPU) with pluggable detection strategies
- NVIDIA Parakeet Vietnamese STT (via NeMo HTTP inference server)
- FastAPI + WebSocket
- Ring buffer + sliding window chunking
- External GPU inference server (NeMo / Ray)

---

# Architecture

## Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Browser / App)                          │
│                  Streams PCM int16 @ 20ms/packet via WebSocket           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ ws://host/ws/stream
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Application (main.py)                      │
│              CORS · WebSocket Router · Health Router                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  WebSocket Router  (/ws/stream)                          │
│  • Connection limit guard (WS_MAX_CONNECTIONS)                           │
│  • Creates StreamingSession per connection                               │
│  • Spawns per-session inference worker (asyncio.Task)                   │
│  • Message dispatch loop                                                 │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ audio packet (base64 PCM)
               ▼
┌─────────────────────────────────────────┐
│           StreamingService               │  ← append to RingAudioBuffer
│  • RingAudioBuffer (12s, np.int16)      │  ← adaptive inference pacing
│  • Snapshot window → audio_queue        │
└─────────────────────────────────────────┘
               │  asyncio.Queue (per session)
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Inference Worker  (per session)                       │
│         (background asyncio.Task, drains audio_queue)                   │
│                                                                          │
│  ┌─────────────┐  probs   ┌───────────────────┐   trimmed audio         │
│  │  SileroVAD  │─────────▶│  Speech Trimmer   │────────────────┐        │
│  │  (ONNX)     │          │  (frame probs)    │                │        │
│  │  VAD Pool   │          └───────────────────┘                │        │
│  └──────┬──────┘                                               │        │
│         │ VADState.update()                                     ▼        │
│         │ silence gate                               ┌──────────────┐   │
│         │                                            │  NeMo ASR    │   │
│         │                                            │  (HTTP/async)│   │
│         │                                            └──────┬───────┘   │
│         │                                                   │ hypothesis │
│         │                                                   ▼            │
│         │                                        ┌────────────────────┐ │
│         │                                        │    Stabilizer      │ │
│         │                                        │  LCP + Rollback    │ │
│         │                                        │  Suppression       │ │
│         │                                        └────────┬───────────┘ │
│         │                                                 │              │
│         │        silence > SILENCE_THRESHOLD_MS           ▼              │
│         └──────────────────────────────────────▶ finalize_transcript    │
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ transcript JSON
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       ConnectionManager                                  │
│  send_transcript(is_final) · send_error · send_backpressure             │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ WebSocket
                                       ▼
                                    CLIENT
```

## Processing Pipeline

Speech-to-text happens in 5 stages. Audio reception and inference run on separate asyncio tasks — the receive loop is never blocked by model latency.

**Stage 1 — Audio buffering**

The client streams 20ms PCM int16 packets at 16kHz over WebSocket. The server appends each packet to a 12-second pre-allocated ring buffer, automatically evicting the oldest samples.

At an adaptive interval (400ms on speech onset, 1200ms when stable), the server snapshots the latest 6 seconds into the session's inference queue and immediately continues receiving. If the queue is full, a `backpressure` message is sent to the client.

**Stage 2 — Voice Activity Detection**

The inference worker runs Silero VAD (ONNX, no PyTorch) on the window to produce per-frame speech probabilities. A trigger strategy (`state_machine` by default) converts these into a binary speech/silence decision, which updates session speaking state.

Frame probabilities are reused in Stage 3 for speech trimming — no second ONNX pass.

**Stage 3 — ASR transcription**

Using VAD frame probabilities, the server trims silence from the inference window, keeping only detected speech plus padding. Trimmed audio shorter than `MIN_TRIMMED_AUDIO_MS` (500ms) is skipped entirely. Otherwise the audio is sent asynchronously to the NVIDIA NeMo HTTP inference server. The response is a **hypothesis** — best guess at that moment, subject to change as more audio arrives.

**Stage 4 — Transcript stabilization**

Streaming ASR produces unstable hypotheses — words at the end may flip or shorten between windows. The stabilizer anchors the agreed prefix via LCP and suppresses rollbacks with the `frozen_prefix` strategy. Only changed text is sent to the client.

**Stage 5 — Finalization**

After 800ms of continuous silence, the current partial is promoted to a final transcript (`is_final: true`). Session state resets for the next utterance.

> For exact function names, variable names, and gate logic: [FLOW.md](FLOW.md)

---

# WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

## Client → Server

| Message | Format |
|---|---|
| Audio packet | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Control | `{"type": "control", "action": "start\|stop"}` |

## Server → Client

| Message | Format |
|---|---|
| Session info | `{"type": "session_info", "session_id": "...", "status": "connected"}` |
| Partial transcript | `{"type": "transcript", "text": "...", "is_final": false}` |
| Final transcript | `{"type": "transcript", "text": "...", "is_final": true}` |
| Backpressure | `{"type": "backpressure", "reason": "queue_full\|vad_pool_exhausted", "dropped_windows": N}` |
| Error | `{"type": "error", "message": "...", "code": "..."}` |

**Control actions:**
- `start` — reset session state (clears buffer, VAD, transcript)
- `stop` — flush any pending partial as a final transcript

---

# HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve static web client (`static/index.html`) |
| `GET` | `/static/*` | Static assets (CSS, JS) |
| `WS` | `/ws/stream` | Streaming audio endpoint |
| `GET` | `/health` | Health check — returns active session count and open WebSocket connection count |

---

# Configuration (`app/core/config.py`)

See [CONFIGURATION.md](../CONFIGURATION.md) for the full parameter reference and priority chain.

---

> For component internals and API reference: [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS.md)
