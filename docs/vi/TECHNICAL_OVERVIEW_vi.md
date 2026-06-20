# Streaming Vietnamese Speech-to-Text — Tổng quan kỹ thuật

Kiến trúc Speech-to-Text streaming đa người dùng sử dụng:

- Silero VAD (CPU) với các chiến lược phát hiện giọng nói có thể thay thế
- NVIDIA Parakeet Vietnamese STT (qua NeMo HTTP inference server)
- FastAPI + WebSocket
- Ring buffer + sliding window chunking
- GPU inference server bên ngoài (NeMo / Ray)

---

# Kiến trúc

## Sơ đồ tổng quan

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
│  • Giới hạn số kết nối (WS_MAX_CONNECTIONS)                             │
│  • Tạo StreamingSession cho mỗi kết nối                                 │
│  • Khởi chạy inference worker riêng (asyncio.Task)                      │
│  • Vòng lặp nhận và phân loại message                                   │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ audio packet (base64 PCM)
               ▼
┌─────────────────────────────────────────┐
│           StreamingService               │  ← ghi vào RingAudioBuffer
│  • RingAudioBuffer (12s, np.int16)      │  ← adaptive inference pacing
│  • Snapshot window → audio_queue        │
└─────────────────────────────────────────┘
               │  asyncio.Queue (mỗi session)
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Inference Worker  (mỗi session)                       │
│         (background asyncio.Task, đọc từ audio_queue)                   │
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
│         │        im lặng > SILENCE_THRESHOLD_MS           ▼              │
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

## Pipeline xử lý

Quá trình chuyển giọng nói thành văn bản diễn ra qua 5 giai đoạn. Việc **nhận âm thanh** và **xử lý inference** chạy trên hai asyncio task riêng biệt — vòng lặp nhận âm thanh không bao giờ bị chặn bởi tốc độ của mô hình AI.

**Giai đoạn 1 — Tích lũy âm thanh**

Client gửi liên tục các gói PCM int16 20ms ở tần số 16kHz qua WebSocket. Server tích lũy từng gói vào ring buffer 12 giây được cấp phát sẵn, tự động ghi đè dữ liệu cũ nhất.

Theo chu kỳ adaptive (400ms khi bắt đầu có tiếng nói, 1200ms khi kết quả ổn định), server cắt 6 giây âm thanh gần nhất vào hàng đợi inference rồi tiếp tục nhận ngay — không chờ kết quả. Nếu hàng đợi đầy, server gửi tín hiệu `backpressure` về client.

**Giai đoạn 2 — Voice Activity Detection**

Inference worker lấy window từ hàng đợi và chạy Silero VAD (ONNX, không cần PyTorch) để tính xác suất có giọng nói cho từng frame. Một trigger strategy (`state_machine` theo mặc định) chuyển dãy xác suất này thành quyết định nhị phân có/không có tiếng nói, cập nhật trạng thái session.

Xác suất từng frame được tái sử dụng ở Giai đoạn 3 để cắt audio — không cần chạy ONNX lần thứ hai.

**Giai đoạn 3 — ASR transcription**

Dựa trên xác suất VAD từng frame, server cắt bỏ phần im lặng ở đầu và cuối window, chỉ giữ lại đoạn có tiếng nói và thêm padding hai bên. Audio ngắn hơn `MIN_TRIMMED_AUDIO_MS` (500ms) sau khi cắt sẽ bị bỏ qua hoàn toàn. Phần còn lại được gửi bất đồng bộ đến NVIDIA NeMo HTTP inference server. Kết quả trả về là một **hypothesis** — kết quả tốt nhất tại thời điểm đó, có thể thay đổi khi nhận thêm âm thanh.

**Giai đoạn 4 — Transcript stabilization**

ASR streaming trả về hypothesis không nhất quán — các từ ở cuối câu có thể thay đổi hoặc bị ngắn lại giữa các window. Stabilizer neo phần prefix đã thống nhất qua thuật toán LCP và chặn rollback bằng chiến lược `frozen_prefix`. Chỉ gửi về client khi văn bản thực sự thay đổi.

**Giai đoạn 5 — Finalization**

Sau 800ms im lặng liên tục, partial transcript hiện tại được xác nhận thành final transcript (`is_final: true`). Trạng thái session được đặt lại để sẵn sàng nhận câu nói tiếp theo.

> Để xem tên hàm, tên biến và logic từng gate cụ thể: [FLOW.md](FLOW_vi.md)

---

# WebSocket Protocol

**Endpoint:** `ws://<host>/ws/stream`

## Client → Server

| Loại message | Format |
|---|---|
| Gói âm thanh | `{"type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000}` |
| Điều khiển | `{"type": "control", "action": "start\|stop"}` |

## Server → Client

| Loại message | Format |
|---|---|
| Thông tin session | `{"type": "session_info", "session_id": "...", "status": "connected"}` |
| Partial transcript | `{"type": "transcript", "text": "...", "is_final": false}` |
| Final transcript | `{"type": "transcript", "text": "...", "is_final": true}` |
| Backpressure | `{"type": "backpressure", "reason": "queue_full\|vad_pool_exhausted", "dropped_windows": N}` |
| Lỗi | `{"type": "error", "message": "...", "code": "..."}` |

**Control actions:**
- `start` — đặt lại trạng thái session (xóa buffer, VAD, transcript)
- `stop` — gửi partial transcript đang có thành final trước khi dừng

---

# HTTP Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Phục vụ web client tích hợp (`static/index.html`) |
| `GET` | `/static/*` | Static assets (CSS, JS) |
| `WS` | `/ws/stream` | Endpoint streaming âm thanh |
| `GET` | `/health` | Health check — trả về số session đang hoạt động và số kết nối WebSocket |

---

# Cấu hình (`app/core/config.py`)

Xem [CONFIGURATION.md](../CONFIGURATION.md) để tham khảo toàn bộ tham số và thứ tự ưu tiên.

---

> Để xem mô tả chi tiết từng component: [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS_vi.md)