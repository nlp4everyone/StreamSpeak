# Mô tả chi tiết các component

## WebSocket Router (`app/routers/websocket_router.py`)

Điểm vào duy nhất cho tất cả kết nối WebSocket. Trách nhiệm:
- Chấp nhận kết nối, tạo session, gửi `session_info`
- Nhận JSON text frame và phân loại theo trường `type`
- `"audio"` → giải mã base64 → `np.int16` → `StreamingHandler.handle_audio_packet()`
- `"control"` → `StreamingHandler.handle_control_message()`
- Khi ngắt kết nối hoặc có lỗi → `StreamingHandler.cleanup_session()`

## StreamingHandler (`app/websocket/handlers.py`)

Điều phối xử lý từng packet — pipeline chính của hệ thống:
- `handle_audio_packet()` — ghi vào buffer, cắt window theo chu kỳ adaptive (`ONSET_INTERVAL_MS`=400ms hoặc `STABLE_INTERVAL_MS`=1200ms) và đưa vào `inference_queue` của session; gửi `backpressure` nếu hàng đợi đầy
- `start_inference_worker()` — khởi chạy một background `asyncio.Task` mỗi session để xử lý `inference_queue` dưới global `inference_semaphore`
- `_run_inference()` — bốn lớp gate trước khi gọi ASR:
  1. **RMS energy gate** — bỏ qua VAD+ASR hoàn toàn khi RMS của window thấp hơn `RMS_SILENCE_THRESHOLD` và session không đang nói; giải phóng VAD pool cho các session đang hoạt động
  2. **VAD gate** — chạy Silero VAD để lấy quyết định có/không có tiếng nói và xác suất từng frame; gửi `backpressure` nếu VAD pool hết chỗ
  3. **Delta gate** — bỏ qua ASR khi `vad_state.last_speech_time` chưa thay đổi so với lần gọi trước (window toàn là im lặng, không có frame nói mới)
  4. **Trim length gate** — sau `_trim_to_speech`, bỏ qua ASR nếu audio sau khi cắt ngắn hơn `MIN_TRIMMED_AUDIO_MS` (500ms); tránh gửi segment gần như im lặng

  Sau đó: STT → stabilize → gửi về client
- `_handle_intra_commit()` — commit partial thành final khi phát hiện khoảng dừng giữa câu (`INTRA_SILENCE_MS`); chỉ kích hoạt một lần mỗi khoảng dừng (được bảo vệ bởi `vad_state.intra_committed`)
- `_trim_to_speech(audio_window, probs)` — cắt window 6 giây về đúng vùng có tiếng nói + padding, tái dùng xác suất frame từ `is_speech()` để tránh chạy ONNX lần thứ hai
- `_extract_final_window()` — trích xuất đoạn audio có boundary chính xác `[speech_start − SPEECH_PADDING_MS, last_speech_time + FINALIZE_RIGHT_PADDING_MS]` từ ring buffer cho final ASR pass
- `_finalize_transcript()` — ghi log số lần ASR call trong lượt nói, tùy chọn chạy final ASR pass qua window chính xác (kết quả đi qua stabilizer để áp dụng frozen prefix và tránh regression), rồi promote partial lên final và gửi `is_final=True`
- `handle_control_message()` — `start` đặt lại trạng thái; `stop` flush partial đang có thành final
- `cleanup_session()` — dừng inference worker, flush partial, xóa session

## ConnectionManager (`app/websocket/manager.py`)

Helper gửi message qua WebSocket cho từng session — `send_transcript()`, `send_error()`, `send_session_info()`, `send_backpressure()`, `connect()`, `disconnect()`.

`send_backpressure(session_id, reason, dropped_windows)` — gửi tín hiệu backpressure về client khi hàng đợi inference đầy (`reason="queue_full"`) hoặc VAD pool timeout (`reason="vad_pool_exhausted"`); được rate-limit ở mức một tín hiệu mỗi giây mỗi session để tránh spam.

## Schema (`app/schema/`)

Pydantic models cho tất cả loại message:
- `websocket.py` — `ErrorMessage`, `ControlMessage`, `SessionInfoMessage`, `WebSocketMessage`
- `audio.py` — schema cho audio message
- `session.py` — schema cho session info
- `transcript.py` — schema cho transcript message
- `health.py` — schema cho health check response

## Quản lý session (`app/session/`)

- `state.py` — `StreamingSession`: sở hữu `RingAudioBuffer`, `VADState`, `TranscriptState`; theo dõi `last_inference_time`, `inference_count`, `last_activity`
- `manager.py` — `SessionManager`: registry session (tạo / lấy / xóa); singleton
- `context.py` — các helper về context session

`VADState` theo dõi `is_speaking`, `speech_start_time`, `last_speech_time`, `silence_duration_ms`, và `intra_committed` (ngăn commit trùng lặp trong cùng một khoảng dừng).

`TranscriptState` sở hữu một instance `BaseStabilizer` **riêng cho mỗi session** (tạo bởi `create_stabilizer()` khi khởi tạo session). Gọi `finalize()` sẽ promote `partial_transcript` thành `final_transcript` và gọi `stabilizer.reset()` để frozen state không lan sang câu nói tiếp theo.

`StreamingSession` còn theo dõi thêm:
- `asr_call_count` — số ASR request trong lượt nói hiện tại; được ghi log khi finalize và reset sau mỗi lượt
- `last_asr_speech_time` — snapshot `vad_state.last_speech_time` tại lần gọi ASR gần nhất; dùng bởi delta gate để bỏ qua call khi không có frame nói mới
- `current_interval_ms` — ngưỡng pacing adaptive của session; bắt đầu ở `ONSET_INTERVAL_MS`, lùi về `STABLE_INTERVAL_MS` khi transcript ổn định, reset khi có tiếng nói mới
- `last_partial_for_stability` — partial transcript gần nhất để phát hiện khi hypothesis ngừng thay đổi

## Ring Buffer (`app/audio/buffer.py`)

Mỗi session giữ tối đa 12 giây PCM samples trong một mảng `np.int16` được cấp phát sẵn (ring buffer thực sự với write-pointer và sample counter). Chỉ chiếm ~384 KB/session — ít hơn 14 lần so với cách tiếp cận `deque`-of-Python-ints trước đó (5.4 MB).

- `append(audio)` — ghi samples mới bằng numpy slice; tự wrap-around, ghi đè dữ liệu cũ nhất
- `get_latest(seconds)` — trích xuất N giây gần nhất thành mảng int16 liên tục (O(N) copy, không dùng Python loop)
- `get_range(start_s, end_s)` — trích xuất một khoảng thời gian cụ thể; dùng bởi `_extract_final_window()` để lấy boundary chính xác
- `clear()` — reset write pointer và counter mà không cấp phát lại bộ nhớ

## Silero VAD (`app/vad/`)

**`silero_vad.py` — `SileroVAD`**

Voice Activity Detection chạy trên CPU qua **ONNX runtime thuần** — không cần PyTorch.
Model được chạy trực tiếp qua `ort.InferenceSession`; GRU hidden state được reset ở đầu mỗi lần inference để các clip độc lập nhau.

Khi khởi động (`app/startup/__init__.py`), một **pool gồm `VAD_POOL_SIZE` (mặc định 8) instance `SileroVAD`** được tạo và đặt vào `asyncio.Queue`. Mỗi lần inference lấy một instance từ pool, chạy ONNX qua `run_in_executor` (thread riêng), rồi trả về — loại bỏ bottleneck `threading.Lock` và cho phép tối đa `VAD_POOL_SIZE` inference VAD đồng thời.

- `is_speech(audio, strategy=...)` — trả về `(decision: bool, probs: list[float])`; caller tái dùng `probs` để cắt audio, tránh chạy ONNX lần thứ hai
- `get_speech_probability(audio)` — xác suất frame cao nhất trong window
- `segments_from_probs(probs, ...)` — tính danh sách `(start_ms, end_ms)` trực tiếp từ danh sách xác suất đã có (dùng bởi `_trim_to_speech`)
- `detect_speech_segments(audio)` — chạy inference rồi gọi `segments_from_probs`; chỉ dùng khi chưa có probs sẵn

**`trigger_strategies.py` — `VADTriggerStrategies`**

| Strategy | Cơ chế |
|---|---|
| `consecutive_frames` | N frame liên tiếp vượt `VAD_THRESHOLD` (mặc định `min_speech_frames=3`) |
| `ema_smoothed` | EMA của xác suất frame > `VAD_THRESHOLD` (`alpha=0.3`) — default ở code level |
| `state_machine` **(default qua settings.yaml)** | FSM với dual-threshold hysteresis: `onset_frames=2` vượt `VAD_ONSET_THRESHOLD` (0.65) để vào trạng thái nói; `offset_frames=3` dưới `VAD_OFFSET_THRESHOLD` (0.40) để thoát — dải trung tính [0.40, 0.65] ngăn chattering |

## NvidiaNemoASREngine (`app/asr/nvidia_nemo/engine.py`)

HTTP client cho NVIDIA NeMo ASR inference server:

```python
POST http://localhost:8005/v1/audio/transcriptions
Content-Type: multipart/form-data
file: audio.wav  (PCM 16-bit, 16kHz, encode trên RAM qua soundfile)
model: nvidia/parakeet-ctc-0.6b-vi
response_format: verbose_json
```

- `transcribe(audio)` — đồng bộ (requests)
- `atranscribe(audio)` — bất đồng bộ, không block; dùng **shared `aiohttp.ClientSession`** để connection pooling qua tất cả inference call, với timeout có thể cấu hình (`ASR_CONNECT_TIMEOUT=2s`, `ASR_REQUEST_TIMEOUT=10s`)
- `aclose()` — đóng shared `ClientSession`; gọi từ `startup.shutdown()` khi tắt máy
- `is_ready()` — GET health probe nhẹ
- Không tạo file tạm; audio được encode vào `BytesIO` trước khi upload

## Transcript Stabilization (`app/stabilization/`)

Làm mịn kết quả ASR streaming không ổn định qua kiến trúc hai lớp:

**Lớp 1 — LCP (Longest Common Prefix)**

`TranscriptStabilizer` (`stabilizer.py`) dùng LCP để neo phần prefix đã ổn định qua các hypothesis liên tiếp. Hai chế độ:

| Chế độ | Khi nào dùng |
|---|---|
| `word_level` (mặc định) | Tiếng Việt và các ngôn ngữ phân tách bằng dấu cách |
| `character_level` | Ngôn ngữ Latin cần độ chính xác cao hơn |

**Lớp 2 — Rollback Suppression**

`BaseStabilizer` (`base.py`) định nghĩa interface cho tất cả chiến lược rollback suppression. Mỗi chiến lược là một stateful object riêng cho từng session; `StabilizationService` không có state và chỉ ủy quyền.

| Chiến lược | Cơ chế | Config |
|---|---|---|
| `frozen_prefix` **(mặc định)** | Đóng băng dần prefix sau khi N hypothesis liên tiếp đồng thuận; từ chối hypothesis mâu thuẫn với vùng đã đóng băng | `STABILIZER_FREEZE_THRESHOLD` |
| `hard_length` | Số từ chỉ được tăng, không giảm | — |
| `edit_distance` | Từ chối hypothesis sai quá N từ so với output gần nhất | `STABILIZER_MAX_EDIT_DISTANCE` |
| `n_consecutive` | Chỉ chấp nhận rollback sau N frame liên tiếp cùng xác nhận | `STABILIZER_N_CONSECUTIVE` |
| `hard_then_frozen` | Pipeline: gate `hard_length` → commit `frozen_prefix` | `STABILIZER_FREEZE_THRESHOLD` |

`StabilizerPipeline` nối nhiều chiến lược theo thứ tự khi `hard_then_frozen` được chọn.

`create_stabilizer()` (`factory.py`) đọc `STABILIZER_STRATEGY` và khởi tạo class tương ứng — thêm chiến lược mới chỉ cần tạo file mới và đăng ký trong factory.

Mỗi session sở hữu stabilizer qua `TranscriptState.stabilizer`; `stabilizer.reset()` được gọi sau mỗi `finalize()` để frozen state không lan sang câu tiếp theo.

```text
hypothesis 1:  xin chào
hypothesis 2:  xin chào m
hypothesis 3:  xin chào mọi
hypothesis 4:  xin chào một      ← rollback → bị chặn bởi frozen_prefix
hypothesis 5:  xin chào mọi người
```

LCP neo phần prefix ổn định; phần cuối không ổn định được thay thế mỗi chu kỳ.
Transcript chỉ được gửi về client khi văn bản sau stabilize thực sự thay đổi so với partial trước — loại bỏ các update không cần thiết.
