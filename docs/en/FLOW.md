# Detailed Inference Flow

## Component Graph

```text
App Startup  (lifespan)
        ├── _maybe_quantize_vad()         VAD_USE_INT8=true → quantize FP32 → INT8 on first run
        ├── SileroVAD × VAD_POOL_SIZE     load ONNX model into pool (asyncio.Queue)
        ├── ThreadPoolExecutor            one thread per VAD instance (true parallelism)
        ├── asyncio.Semaphore             global ASR cap (ASR_SEMAPHORE_LIMIT=8)
        └── asyncio.Task: idle-cleanup    runs every 60 s; closes sessions idle > 300 s

Client (Browser / App)
        │  JSON over WebSocket
        │  {"type": "audio", "data": "<base64 PCM>"}
        ▼
FastAPI WebSocket Gateway  (/ws/stream)
        │  connection_count >= WS_MAX_CONNECTIONS?
        │      YES → accept() + close(1013, "server_full") → return   ← no state allocated
        │      NO  ↓
        ├── ConnectionManager  (per-session WS send helpers; singleton)
        ├── SessionManager     (session registry)
        └── StreamingHandler   (per-packet orchestration)
                │
                ▼
        StreamingSession  (per-session state)
                ├── RingAudioBuffer  (12s ring buffer, pre-allocated np.int16)
                ├── VADState         (speaking / silence / intra-commit tracking)
                ├── TranscriptState  (partial / final transcript + per-session stabilizer)
                └── inference_queue  (asyncio.Queue, maxsize=INFERENCE_QUEUE_MAXSIZE)
                │
                │  handle_audio_packet() enqueues audio_snapshot at adaptive interval
                │    (ONSET_INTERVAL_MS=400ms on new speech; STABLE_INTERVAL_MS=1200ms when transcript stable)
                │  _inference_worker() drains queue per session
                ▼
        StreamingHandler._inference_worker()  [background asyncio.Task per session]
                │  async with inference_semaphore  (ASR_SEMAPHORE_LIMIT global cap)
                ▼
        StreamingHandler._run_inference()
                │
                ├──▶ RMS energy gate  [if NOT is_speaking]
                │       rms < RMS_SILENCE_THRESHOLD (300)?  → skip entirely (frees VAD pool)
                │
                ├──▶ VAD pool (asyncio.Queue of VAD_POOL_SIZE SileroVAD instances)
                │       asyncio.wait_for(pool.get(), timeout=5.0)
                │         timeout → drop window + backpressure (rate-limited 1/s)
                │       run_in_executor → vad.is_speech()   (dedicated thread per instance)
                │       returns (decision: bool, probs: list[float])
                │       release instance back to pool
                │         └── VADTriggerStrategies
                │               (consecutive_frames | ema_smoothed | state_machine)
                │
                ├──▶ _handle_intra_commit()   [if INTRA_SILENCE_COMMIT_ENABLED]
                │       fires once per pause when: is_speaking=True AND
                │       silence_duration >= INTRA_SILENCE_MS AND not intra_committed
                │       → TranscriptState.finalize() + send_transcript(is_final=True)
                │
                │  if speech detected (decision OR vad_state.is_speaking):
                │    └──▶ Delta gate: last_speech_time unchanged since last ASR?  → skip ASR
                ▼
        StreamingHandler._trim_to_speech(audio_window, probs)
                │  SileroVAD.segments_from_probs(probs)   ← reuses VAD probs, no 2nd ONNX pass
                │  crop to [first_start − padding, last_end + padding]
                │  falls back to full window if no segments found
                │  trimmed_length < MIN_TRIMMED_AUDIO_MS (500 ms)?  → skip ASR
                ▼
        TranscriptionService.atranscribe(trimmed_audio)  [async, asr_call_count++]
          └── NvidiaNemoASREngine.atranscribe(audio)
                    │  encode as in-memory WAV (soundfile, PCM 16-bit)
                    │  shared aiohttp.ClientSession POST multipart/form-data
                    ▼
            NeMo Inference Server
            nvidia/parakeet-ctc-0.6b-vi
                    │
                    ▼
                raw transcript text
                │
                │  only if stabilized text differs from previous partial
                ▼
        StabilizationService.stabilize(session.stabilizer, new_hypothesis)
          └── BaseStabilizer  (per-session, created by factory.create_stabilizer())
                │  strategies: frozen_prefix | hard_length | edit_distance |
                │               n_consecutive | hard_then_frozen
                │  mode: word_level (default) | character_level
                ▼
        ConnectionManager.send_transcript()
                │
                ▼
Client  ← {"type": "transcript", "text": "...", "is_final": false|true}

        │
        │  if NOT vad_state.is_speaking AND partial_transcript exists
        ▼
        StreamingHandler._finalize_transcript()
                │  if FINALIZE_RIGHT_PADDING_ENABLED:
                │      _extract_final_window()   ← precise window:
                │          [speech_start - SPEECH_PADDING_MS,
                │           last_speech_time + FINALIZE_RIGHT_PADDING_MS]
                │      atranscribe(final_window)  ← dedicated ASR pass
                │      overrides partial if result non-empty
                │  TranscriptState.finalize()
                └── send_transcript(is_final=True)

App Shutdown  (SIGTERM / lifespan exit)
        ├── cancel idle-cleanup task
        ├── _stop_inference_worker()  ×  all active sessions   (parallel)
        ├── _finalize_transcript()   ×  all active sessions   (parallel, 15 s timeout)
        ├── TranscriptionService.aclose()   ← close shared aiohttp session
        └── vad_executor.shutdown()
```

---

## Step-by-Step Flow

```text
⓪ App startup  (lifespan)
    │  _maybe_quantize_vad()    if VAD_USE_INT8=true AND _int8.onnx missing → quantize FP32 model
    │  SileroVAD × 8            load VAD_POOL_SIZE instances into asyncio.Queue (vad_pool)
    │  ThreadPoolExecutor       max_workers=VAD_POOL_SIZE, thread_name_prefix="vad"
    │  asyncio.Semaphore        inference_semaphore (ASR_SEMAPHORE_LIMIT=8)
    │  asyncio.Task             idle-cleanup loop (every 60 s, timeout 300 s)
    ▼
① Client sends 20ms PCM packets  (base64 JSON, 16kHz int16)
    │
    ▼
② WebSocket route  (app/routers/websocket_router.py)
    │
    │  capacity check:
    │      connection_count >= WS_MAX_CONNECTIONS (200)?
    │          YES → accept() + close(code=1013, reason="server_full") → return
    │          NO  → create session + accept connection + send session_info
    │
    │  start_inference_worker(session)   ← spawn background asyncio.Task per session
    │
    │  message loop:
    │      receive_text() → JSON parse → session.update_activity()  ← ALL message types
    │      "audio"   → base64 decode → np.frombuffer(dtype=np.int16) → handle_audio_packet()
    │      "control" → handle_control_message(action)
    │
    ▼
③ StreamingHandler.handle_audio_packet()
    │
    ├─ StreamingService.process_audio_packet()
    │       RingAudioBuffer.append(packet)   ← np.int16 ring buffer, auto-evict oldest
    │
    └─ StreamingService.should_run_inference()
            elapsed >= session.current_interval_ms?   ← adaptive: 400ms (onset) or 1200ms (stable)
                NO  → return  (wait for next packet)   ← falls back to INFERENCE_INTERVAL_MS (600ms) if ADAPTIVE_INTERVAL_ENABLED=false
               YES  → get_inference_window()  → last INFERENCE_WINDOW_SECONDS (6 s) of audio
                        session.audio_queue.put_nowait(window)
                          QueueFull?  → dropped_windows++
                                        send backpressure (rate-limited 1/s): reason="queue_full"
    │
    ▼
④ StreamingHandler._inference_worker()  [background asyncio.Task per session]
    │  async with inference_semaphore  ← global cap (ASR_SEMAPHORE_LIMIT=8)
    ▼
    StreamingHandler._run_inference(audio_window)
    │
    ├─ RMS energy gate  [if NOT vad_state.is_speaking]
    │       rms = sqrt(mean(audio_window²))
    │       rms < RMS_SILENCE_THRESHOLD (300)?  → skip VAD+ASR entirely — return
    │           ← frees the VAD pool for sessions that are actively speaking
    │
    ├─ _run_vad(session, audio_window)
    │       asyncio.wait_for(vad_pool.get(), timeout=5.0)
    │           TimeoutError?  → dropped_windows++
    │                            send backpressure: reason="vad_pool_exhausted"
    │                            return (False, [])
    │       loop.run_in_executor(vad_executor, vad.is_speech, audio_window, strategy)
    │           ← runs on dedicated VAD thread; event loop stays unblocked
    │           ← GRU hidden state reset per call (clips are independent)
    │       vad_pool.put_nowait(vad)   ← release immediately after inference
    │       returns (decision: bool, probs: list[float])
    │
    │   Trailing-silence window correction  [before VADState.update]
    │       is_speech=True AND last speech segment ended >= TRAILING_SILENCE_MS (1000ms) ago?
    │           YES → override is_speech=False  ← prevents stale VAD; ~50% fewer ASR calls at utterance end
    │
    │   VADState.update(decision, now)
    │       silence_duration >= SILENCE_THRESHOLD_MS (800 ms)  →  is_speaking = False
    │
    ├─ _handle_intra_commit()   [if INTRA_SILENCE_COMMIT_ENABLED=True]
    │       fires when: is_speaking=True
    │                   AND silence_duration_ms >= INTRA_SILENCE_MS (300 ms)
    │                   AND NOT vad_state.intra_committed        ← once per pause
    │                   AND partial_transcript non-empty
    │       → vad_state.intra_committed = True
    │       → TranscriptState.finalize()
    │       → send_transcript(is_final=True)   ← mid-sentence segment committed
    │       intra_committed resets to False on next speech frame
    │
    │   if NOT (decision OR vad_state.is_speaking):
    │       skip STT  ──────────────────────────────────────────────────────┐
    │                                                                        │
    ├─ Delta gate  [if is_speech OR is_speaking]                             │
    │       current_speech_ts = vad_state.last_speech_time                  │
    │       current_speech_ts == session.last_asr_speech_time?               │
    │           YES → no new speech frames; skip ASR, fall through to ⑧    │
    │           NO  → update last_asr_speech_time                           │
    │                 adaptive interval step 1: reset to ONSET_INTERVAL_MS  │
    │                                                                        │
    ├─ _trim_to_speech(audio_window, probs)                                  │
    │       self._vad_ref.segments_from_probs(probs)                         │
    │           ← pure Python; reuses existing probs — no 2nd ONNX pass     │
    │       start = max(0, first_segment_start_ms / 1000 × sr − padding)    │
    │       end   = min(len, last_segment_end_ms   / 1000 × sr + padding)   │
    │       padding = SPEECH_PADDING_MS (200 ms) = 3200 samples @ 16 kHz   │
    │       no segments found?  → use full audio_window as fallback          │
    │       trimmed_length < MIN_TRIMMED_AUDIO_MS (500 ms)?  → skip ASR     │
    │                                                                        │
    ├─ TranscriptionService.atranscribe(trimmed_audio)  [asr_call_count++]  │
    │       NvidiaNemoASREngine.atranscribe()                                │
    │           soundfile → in-memory BytesIO WAV (PCM 16-bit, mono)        │
    │           shared aiohttp.ClientSession POST multipart/form-data        │
    │           → NEMO_API_URL /v1/audio/transcriptions                      │
    │           connect_timeout=ASR_CONNECT_TIMEOUT (2 s)                   │
    │           total_timeout=ASR_REQUEST_TIMEOUT (10 s)                     │
    │           response["text"]                                             │
    │                                                                        │
    ├─ StabilizationService.stabilize(session.stabilizer, new_hypothesis)   │
    │       delegates to per-session BaseStabilizer                          │
    │       strategy selected at session creation via create_stabilizer()    │
    │       stabilizer.reset() called after each utterance finalize          │
    │                                                                        │
    ├─ if stabilized != previous_partial:                                    │
    │       TranscriptState.update_partial(stabilized)                      │
    │       ConnectionManager.send_transcript(is_final=False)                │
    │                                                                        │
    │   adaptive interval step 2: transcript unchanged → STABLE_INTERVAL_MS │
    │                             transcript changed   → ONSET_INTERVAL_MS  │
    │                                               ◄────────────────────────┘
⑧  └─ if NOT vad_state.is_speaking AND partial_transcript exists:
            _finalize_transcript()
                log asr_call_count for this turn; reset counter
                if FINALIZE_RIGHT_PADDING_ENABLED=True:
                    _extract_final_window()
                        end_ago   = now − last_speech_time − FINALIZE_RIGHT_PADDING_MS
                        start_ago = now − speech_start_time + SPEECH_PADDING_MS
                        RingAudioBuffer.get_range(start_ago, end_ago)
                    atranscribe(final_window)   ← dedicated ASR pass, precise boundaries
                    stabilize(final_window)     ← frozen prefix applied; prevents "Unk" regressions
                    overrides partial if result non-empty
                TranscriptState.finalize()       ← partial → final; stabilizer.reset()
                send_transcript(is_final=True)
    │
    ▼
⑤ Disconnect / idle-timeout / cleanup
    WebSocketDisconnect (client drop or idle_timeout close by server)
    or unhandled server error
    →  StreamingHandler.cleanup_session()
            _stop_inference_worker()    cancel + await task
            _finalize_transcript()      flush pending partial as final (if any)
            SessionManager.remove_session()
            ConnectionManager.disconnect()

⑥ Graceful shutdown  (SIGTERM / docker stop / deploy)
    startup.shutdown()
        cancel idle-cleanup task
        for each active session (parallel):
            _stop_inference_worker()    ← stop first; no new partials during finalize
        for each active session (parallel, 15 s timeout):
            _finalize_transcript()      ← send pending partial as final to all clients
        TranscriptionService.aclose()  ← close shared aiohttp ClientSession
        vad_executor.shutdown()
```

---

Inference windows overlap to preserve speech context across packet boundaries:

```text
t=0.0s  [0.0s → 6.0s]
t=0.4s  [0.4s → 6.4s]
t=0.8s  [0.8s → 6.8s]
```
