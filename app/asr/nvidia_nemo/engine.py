import numpy as np
import soundfile as sf
from app.core.config import settings
import requests, aiohttp, io

class NvidiaNemoASREngine:
    """
    HTTP client for an Nvidia NeMo ASR inference server.

    Sends audio to a locally-hosted NeMo server that exposes an
    OpenAI-compatible ``POST /v1/audio/transcriptions`` endpoint.
    Audio is encoded as a 16-bit PCM WAV in memory before being sent
    as a multipart form upload; the server returns a JSON body whose
    ``text`` field contains the transcript.

    Typical deployment: the NeMo server runs in a separate container
    (default port 8005) and serves a Vietnamese CTC model
    (``nvidia/parakeet-ctc-0.6b-vi``).
    """

    def __init__(self,
                 api_url: str = settings.NEMO_API_URL,
                 model: str = settings.NEMO_MODEL,
                 sample_rate: int = settings.SAMPLE_RATE):
        """
        Args:
            api_url: Full URL of the NeMo transcription endpoint.
                Defaults to ``settings.NEMO_API_URL`` (env: ``NEMO_API_URL``).
            model: Model identifier forwarded to the server in the
                ``model`` form field.
                Defaults to ``settings.NEMO_MODEL`` (env: ``NEMO_MODEL``).
            sample_rate: Sample rate of all audio passed to
                :meth:`transcribe`. Must match what the model expects
                (16 kHz for Parakeet).
        """
        self.api_url = api_url
        self.model = model
        self.sample_rate = sample_rate

    def _to_wav_bytes(self,
                      audio: np.ndarray) -> io.BytesIO:
        """
        Encode *audio* as a 16-bit PCM WAV in an in-memory buffer.

        If *audio* is float32 (assumed range [-1.0, 1.0]), it is scaled
        and clipped to int16 before encoding. Clipping guards against
        values that slightly exceed ±1.0 due to upstream processing.

        Args:
            audio: 1-D numpy array, either int16 or float32.

        Returns:
            Seeked-to-start ``BytesIO`` containing the WAV file bytes.
        """
        # Float32 input is assumed to be in [-1.0, 1.0]; scale to full int16
        # range and clip to guard against values that slightly exceed ±1.0
        # (common after mixing or resampling).
        if audio.dtype != np.int16:
            audio = (audio * 32768).clip(-32768, 32767).astype(np.int16)

        # Write directly to an in-memory buffer — no temporary file on disk.
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)  # rewind so the caller reads from the beginning
        return buffer

    def transcribe(self,
                   audio: np.ndarray) -> str:
        """
        Transcribe *audio* to text via the NeMo ASR server.

        The audio is encoded as WAV and posted to the server as a
        multipart form upload. The server is expected to return JSON
        with ``response_format="verbose_json"``.

        Args:
            audio: 1-D numpy array of audio samples (int16 or float32)
                at :attr:`sample_rate` Hz.

        Returns:
            Transcript string, or ``""`` if the server returns no text.

        Raises:
            requests.HTTPError: If the server returns a non-2xx status.
            requests.RequestException: On connection or timeout errors.
        """
        # Step 1: encode numpy audio to 16-bit PCM WAV in memory.
        wav_bytes = self._to_wav_bytes(audio)

        # Step 2: POST to the NeMo server as a multipart form upload.
        # "verbose_json" makes the server include word-level timestamps in
        # the response body, which we may use for alignment in the future.
        response = requests.post(
            self.api_url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={
                "model": self.model,
                "response_format": "verbose_json",
            },
        )

        # Step 3: raise immediately on HTTP errors (4xx / 5xx) so the
        # caller can decide how to retry or surface the failure.
        response.raise_for_status()

        # Step 4: extract the transcript; fall back to "" if the key is
        # absent (e.g. silence or unsupported audio).
        return response.json().get("text", "")

    async def atranscribe(self,
                          audio: np.ndarray) -> str:
        """
        Async version of :meth:`transcribe` using ``aiohttp``.

        Encodes *audio* as WAV and POSTs it to the NeMo server without
        blocking the event loop. Prefer this method inside ``async``
        request handlers or WebSocket coroutines; use :meth:`transcribe`
        only in synchronous contexts.

        Args:
            audio: 1-D numpy array of audio samples (int16 or float32)
                at :attr:`sample_rate` Hz.

        Returns:
            Transcript string, or ``""`` if the server returns no text.

        Raises:
            aiohttp.ClientResponseError: If the server returns a non-2xx
                status (equivalent of ``raise_for_status``).
            aiohttp.ClientError: On connection or timeout errors.
        """
        # Step 1: encode numpy audio to 16-bit PCM WAV in memory.
        # _to_wav_bytes is CPU-bound but fast (<1 ms for typical chunks),
        # so running it inline is fine without run_in_executor.
        wav_bytes = self._to_wav_bytes(audio)

        # Step 2: build the multipart form payload.
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        form.add_field("response_format", "verbose_json")

        # Step 3: POST asynchronously; a new ClientSession per call keeps
        # the engine stateless — no shared session to manage or close.
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, data=form) as response:
                # Step 4: raise on HTTP errors (4xx / 5xx).
                response.raise_for_status()

                # Step 5: parse JSON and extract transcript.
                body = await response.json(content_type=None)
                return body.get("text", "")

    def is_ready(self) -> bool:
        """
        Probe the ASR server with a lightweight GET request.

        Returns:
            ``True`` if the server responds within 2 seconds,
            ``False`` on any connection or timeout error.
        """
        try:
            # A GET to the transcription endpoint is enough to confirm the
            # server is up; we don't parse the response body.
            requests.get(self.api_url, timeout=2)
            return True
        except Exception:
            # Covers ConnectionError, Timeout, and any other network issue.
            return False
