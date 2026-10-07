#
# Custom Pipecat TTS service wrapping Kyutai's delayed-streams-modeling TTS
# (moshi_mlx), for on-device, exact-text speech synthesis on Apple Silicon.
#
# Follows the same shape as pipecat's built-in local-model services (e.g.
# pipecat.services.kokoro.tts.KokoroTTSService): load the model once in
# __init__, then stream PCM frames out of run_tts().
#
# moshi_mlx's TTSModel.generate() is synchronous and blocking, so it's run
# in a worker thread; its on_frame callback pushes decoded PCM chunks into
# a thread-safe queue that the async generator drains, which is what gives
# accurate TTFB (time to first audio byte) metrics and real streaming
# instead of buffering the whole utterance before yielding anything.
#

import asyncio
import json
import queue
import threading
import time
from collections.abc import AsyncGenerator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
from loguru import logger
from moshi_mlx import models
from moshi_mlx.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
from moshi_mlx.utils.loaders import hf_get

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KyutaiTTSService(TTSService):
    """On-device TTS using Kyutai's delayed-streams-modeling (moshi_mlx).

    Synthesizes exact input text deterministically -- unlike a conversational
    speech-to-speech model, there's no risk of paraphrasing or refusing. Good
    fit for speaking fixed strings such as tool-call results.
    """

    def __init__(
        self,
        *,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
        quantize: int | None = 8,
        stop_frame_timeout_s: float = 20.0,
        **kwargs,
    ):
        # MLX pays a one-time Metal shader compilation cost on a model's first
        # inference call in a process, on top of the model's own (slow, ~0.5x
        # real-time) generation speed. The base class's default 3s
        # stop_frame_timeout_s is tuned for cloud TTS and tears the turn down
        # as "no audio" long before this on-device model produces a first
        # chunk, so it needs a much more generous budget here.
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=stop_frame_timeout_s,
            **kwargs,
        )

        self._voice = voice
        mx.random.seed(299792458)

        logger.info("KyutaiTTS: retrieving checkpoints")
        raw_config = hf_get("config.json", hf_repo)
        with open(hf_get(raw_config), "r") as fobj:
            raw_config = json.load(fobj)

        mimi_weights = hf_get(raw_config["mimi_name"], hf_repo)
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, hf_repo)
        tokenizer = hf_get(raw_config["tokenizer_name"], hf_repo)
        lm_config = models.LmConfig.from_config_dict(raw_config)
        # Workaround for a ring kv cache bug in moshi_mlx <= 0.3.0.
        lm_config.transformer.max_seq_len = lm_config.transformer.context

        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        logger.info(f"KyutaiTTS: loading model weights from {moshi_weights}")
        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if quantize is not None:
            logger.info(f"KyutaiTTS: quantizing model to {quantize} bits")
            nn.quantize(model.depformer, bits=quantize)
            for layer in model.transformer.layers:
                nn.quantize(layer.self_attn, bits=quantize)
                nn.quantize(layer.gating, bits=quantize)

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))  # type: ignore

        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        self._tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=voice_repo,
            temp=0.6,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=raw_config,
        )

        if self._tts_model.valid_cfg_conditionings:
            self._cfg_coef_conditioning = self._tts_model.cfg_coef
            self._tts_model.cfg_coef = 1.0
            self._cfg_is_no_text = False
            self._cfg_is_no_prefix = False
        else:
            self._cfg_coef_conditioning = None
            self._cfg_is_no_text = True
            self._cfg_is_no_prefix = True

        self._mimi_sample_rate = self._tts_model.mimi.sample_rate
        self._resampler = create_stream_resampler()

        logger.info("KyutaiTTS: warming up (first-call Metal/JIT compilation)...")
        warmup_begin = time.time()
        self._generate_sync("Ready.")
        logger.info(f"KyutaiTTS: warm-up took {time.time() - warmup_begin:.2f}s, model ready")

    def _generate_sync(self, text: str, on_frame=lambda frame: None):
        """Run the blocking moshi_mlx generation loop for a single utterance."""
        all_entries = [self._tts_model.prepare_script([text])]
        voices = (
            [self._tts_model.get_voice_path(self._voice)] if self._tts_model.multi_speaker else []
        )
        all_attributes = [
            self._tts_model.make_condition_attributes(voices, self._cfg_coef_conditioning)
        ]
        self._tts_model.generate(
            all_entries,
            all_attributes,
            cfg_is_no_prefix=self._cfg_is_no_prefix,
            cfg_is_no_text=self._cfg_is_no_text,
            on_frame=on_frame,
        )

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)

            loop = asyncio.get_running_loop()
            pcm_queue: queue.Queue = queue.Queue()
            done = threading.Event()
            error_holder: list[Exception] = []

            def on_frame(frame):
                if (frame == -1).any():
                    return
                pcm = self._tts_model.mimi.decode_step(frame[:, :, None])
                pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
                pcm_queue.put_nowait(pcm)

            def generate():
                try:
                    begin = time.time()
                    self._generate_sync(text, on_frame=on_frame)
                    logger.debug(f"KyutaiTTS: generation took {time.time() - begin:.2f}s")
                except Exception as e:  # noqa: BLE001
                    error_holder.append(e)
                finally:
                    done.set()

            worker = threading.Thread(target=generate, daemon=True)
            worker.start()

            first_chunk = True
            while not (done.is_set() and pcm_queue.empty()):
                try:
                    pcm = await loop.run_in_executor(
                        None, pcm_queue.get, True, 0.05
                    )
                except queue.Empty:
                    continue

                if first_chunk:
                    await self.stop_ttfb_metrics()
                    first_chunk = False

                audio_int16 = (pcm * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, self._mimi_sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )

            worker.join()
            if error_holder:
                raise error_holder[0]

        except Exception as e:  # noqa: BLE001
            logger.exception("KyutaiTTS: error during synthesis")
            yield ErrorFrame(error=f"KyutaiTTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
