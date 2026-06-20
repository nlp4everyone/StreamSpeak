# Core Components

## WebSocket Router (`app/routers/websocket_router.py`)

Entry point for all WebSocket connections. Responsibilities:
- Accept connection, create session, send `session_info`
- Receive JSON text frames and dispatch by `type`
- `"audio"` → base64 decode → `np.int16` → `StreamingHandler.handle_audio_packet()`
- `"control"` → `StreamingHandler.handle_control_message()`
- On disconnect or error → `StreamingHandler.cleanup_session()`

## StreamingHandler (`app/websocket/handlers.py`)

Per-packet orchestration — the main processing pipeline:
- `handle_audio_packet()` — appends to buffer, snapshots audio window at the current adaptive interval (`ONSET_INTERVAL_MS`=400ms or `STABLE_INTERVAL_MS`=1200ms) and enqueues to the session's `inference_queue`; sends `backpressure` if queue is full
- `start_inference_worker()` — spawns a background `asyncio.Task` per session that drains `inference_queue` under the global `inference_semaphore`
- `_run_inference()` — four-layer gate before ASR:
  1. **RMS energy gate** — skips VAD+ASR entirely when the window RMS is below `RMS_SILENCE_THRESHOLD` and the session is not mid-utterance; frees the shared VAD pool for active sessions
  2. **VAD gate** — runs Silero VAD to get speech decision and per-frame probabilities; sends `backpressure` if VAD pool is exhausted
  3. **Delta gate** — skips ASR when `vad_state.last_speech_time` hasn't advanced since the previous call (window is pure silence, no new speech frames)
  4. **Trim length gate** — after `_trim_to_speech`, skips ASR if trimmed audio is shorter than `MIN_TRIMMED_AUDIO_MS` (500ms); avoids sending near-silent segments

  Then: STT → stabilize → send
- `_handle_intra_commit()` — commits the current partial as final on mid-utterance pauses (`INTRA_SILENCE_MS`); fires once per pause (guarded by `vad_state.intra_committed`)
- `_trim_to_speech(audio_window, probs)` — crops the 6s inference window to the detected speech region + padding using frame probs already computed by `is_speech()`, avoiding a second ONNX pass
- `_extract_final_window()` — extracts a precisely-bounded audio slice `[speech_start − SPEECH_PADDING_MS, last_speech_time + FINALIZE_RIGHT_PADDING_MS]` from the ring buffer for the final ASR pass
- `_finalize_transcript()` — logs ASR call count for the turn, optionally runs a dedicated final ASR pass over the precise speech window (result is passed through the stabilizer to apply the frozen prefix and prevent raw ASR regressions), then promotes the partial to final and sends `is_final=True`
- `handle_control_message()` — `start` resets state; `stop` flushes pending partial as final
- `cleanup_session()` — stops inference worker, flushes pending partial, removes session

## ConnectionManager (`app/websocket/manager.py`)

Per-session WebSocket send helpers — `send_transcript()`, `send_error()`, `send_session_info()`, `send_backpressure()`, `connect()`, `disconnect()`.

`send_backpressure(session_id, reason, dropped_windows)` — sends a backpressure signal to the client when either the inference queue is full (`reason="queue_full"`) or the VAD pool times out (`reason="vad_pool_exhausted"`); rate-limited to one signal per second per session to avoid flooding.

## Schema (`app/schema/`)

Pydantic models for all message types:
- `websocket.py` — `ErrorMessage`, `ControlMessage`, `SessionInfoMessage`, `WebSocketMessage`
- `audio.py` — audio message schema
- `session.py` — session info schema
- `transcript.py` — transcript message schema
- `health.py` — health check response schema

## Session Management (`app/session/`)

- `state.py` — `StreamingSession`: owns `RingAudioBuffer`, `VADState`, `TranscriptState`; tracks `last_inference_time`, `inference_count`, `last_activity`
- `manager.py` — `SessionManager`: session registry (create / get / remove); singleton
- `context.py` — session context helpers

`VADState` tracks `is_speaking`, `speech_start_time`, `last_speech_time`, `silence_duration_ms`, and `intra_committed` (prevents duplicate intra-utterance commits per pause).

`TranscriptState` owns a **per-session `BaseStabilizer`** instance (created by `create_stabilizer()` at session construction). Calling `finalize()` promotes `partial_transcript` to `final_transcript` and calls `stabilizer.reset()` so frozen state from this utterance does not bleed into the next.

`StreamingSession` additionally tracks:
- `asr_call_count` — ASR requests made during the current speech turn; logged on finalize and reset each turn
- `last_asr_speech_time` — `vad_state.last_speech_time` snapshot at the last ASR call; used by the delta gate to skip calls where no new speech frames arrived
- `current_interval_ms` — per-session adaptive pacing threshold; starts at `ONSET_INTERVAL_MS`, backs off to `STABLE_INTERVAL_MS` when transcript is stable, resets on new speech
- `last_partial_for_stability` — last stabilized transcript used to detect when the hypothesis has stopped changing

## Ring Buffer (`app/audio/buffer.py`)

Each session holds up to 12 seconds of PCM samples in a pre-allocated `np.int16` array (a true ring buffer with a write-pointer and sample counter). This uses ~384 KB/session — 14× less than the previous `deque`-of-Python-ints approach (5.4 MB).

- `append(audio)` — push new samples using numpy slice writes; wraps around automatically, evicting oldest
- `get_latest(seconds)` — extract the most-recent N seconds as a contiguous int16 ndarray (O(N) copy, no Python loops)
- `get_range(start_s, end_s)` — extract a specific time slice; used by `_extract_final_window()` for precise speech boundary extraction
- `clear()` — reset write pointer and count without reallocating

## Silero VAD (`app/vad/`)

**`silero_vad.py` — `SileroVAD`**

CPU-based Voice Activity Detection running via **pure ONNX runtime** — no PyTorch dependency.
The model is driven directly via `ort.InferenceSession`; GRU hidden state is reset at the start
of each inference call to keep clips independent.

At app startup (`app/startup/__init__.py`), a **pool of `VAD_POOL_SIZE` (default 8) `SileroVAD`
instances** is created and placed in an `asyncio.Queue`. Each inference call acquires an instance
from the pool, runs ONNX inference via `run_in_executor` (dedicated thread), then releases the
instance back — eliminating the `threading.Lock` bottleneck and allowing up to `VAD_POOL_SIZE`
concurrent VAD inferences.

- `is_speech(audio, strategy=...)` — returns `(decision: bool, probs: list[float])`; callers reuse `probs` for speech trimming to avoid a redundant ONNX pass
- `get_speech_probability(audio)` — peak frame probability across the window
- `segments_from_probs(probs, ...)` — derive `(start_ms, end_ms)` speech segments directly from a pre-computed probability list (used by `_trim_to_speech`)
- `detect_speech_segments(audio)` — runs inference then calls `segments_from_probs`; use only when probs are not already available

**`trigger_strategies.py` — `VADTriggerStrategies`**

| Strategy | Mechanism |
|---|---|
| `consecutive_frames` | N consecutive frames above `VAD_THRESHOLD` (default `min_speech_frames=3`) |
| `ema_smoothed` | EMA of frame probs > `VAD_THRESHOLD` (`alpha=0.3`) — code-level default |
| `state_machine` **(default via settings.yaml)** | FSM with dual-threshold hysteresis: `onset_frames=2` above `VAD_ONSET_THRESHOLD` (0.65) to enter speech; `offset_frames=3` below `VAD_OFFSET_THRESHOLD` (0.40) to exit — the neutral band [0.40, 0.65] prevents chattering |

## NvidiaNemoASREngine (`app/asr/nvidia_nemo/engine.py`)

HTTP client for an NVIDIA NeMo ASR inference server:

```python
POST http://localhost:8005/v1/audio/transcriptions
Content-Type: multipart/form-data
file: audio.wav  (PCM 16-bit, 16kHz, encoded in-memory via soundfile)
model: nvidia/parakeet-ctc-0.6b-vi
response_format: verbose_json
```

- `transcribe(audio)` — synchronous (requests)
- `atranscribe(audio)` — async, non-blocking; uses a **shared `aiohttp.ClientSession`** for connection pooling across all inference calls, with configurable timeouts (`ASR_CONNECT_TIMEOUT=2s`, `ASR_REQUEST_TIMEOUT=10s`)
- `aclose()` — closes the shared `ClientSession`; called from `startup.shutdown()` for clean teardown
- `is_ready()` — lightweight GET health probe
- No temp files; audio is encoded to `BytesIO` before upload

## Transcript Stabilization (`app/stabilization/`)

Smooths unstable streaming ASR output using a two-layer architecture:

**Layer 1 — LCP (Longest Common Prefix)**

`TranscriptStabilizer` (`stabilizer.py`) uses LCP to anchor the stable prefix across consecutive hypotheses. Two modes:

| Mode | When to use |
|---|---|
| `word_level` (default) | Vietnamese and other space-delimited scripts |
| `character_level` | Latin-script languages needing finer precision |

**Layer 2 — Rollback Suppression**

`BaseStabilizer` (`base.py`) defines the interface for all rollback suppression strategies. Each is a per-session stateful object; `StabilizationService` is stateless and just delegates.

| Strategy | Mechanism | Config |
|---|---|---|
| `frozen_prefix` **(default)** | Progressively freezes a prefix once N consecutive hypotheses agree; rejects any hypothesis that contradicts the frozen region | `STABILIZER_FREEZE_THRESHOLD` |
| `hard_length` | Monotonic word-count guard — transcript length can only grow, never shrink | — |
| `edit_distance` | Rejects hypotheses that deviate more than N word edits from the last accepted output | `STABILIZER_MAX_EDIT_DISTANCE` |
| `n_consecutive` | Accepts rollbacks only after N consecutive frames all show the shorter hypothesis | `STABILIZER_N_CONSECUTIVE` |
| `hard_then_frozen` | Pipeline: `hard_length` gate → `frozen_prefix` commit | `STABILIZER_FREEZE_THRESHOLD` |

`StabilizerPipeline` chains multiple strategies left-to-right when `hard_then_frozen` is selected.

`create_stabilizer()` (`factory.py`) reads `STABILIZER_STRATEGY` and instantiates the correct class — adding a new strategy requires only a new file and a factory entry.

Each session owns its stabilizer via `TranscriptState.stabilizer`; `stabilizer.reset()` is called after every `finalize()` so frozen state does not carry over between utterances.

```text
hypothesis 1:  xin chào
hypothesis 2:  xin chào m
hypothesis 3:  xin chào mọi
hypothesis 4:  xin chào một      ← rollback → suppressed by frozen_prefix
hypothesis 5:  xin chào mọi người
```

LCP anchors the stable prefix; the unstable suffix is replaced each cycle.
The transcript is only sent to the client when the stabilized text actually differs from the
previous partial — suppressing no-op updates.
