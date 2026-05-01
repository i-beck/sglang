"""Unit tests for key Nano Nemotron VL processor audio paths."""

import asyncio
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.nano_nemotron_vl import NanoNemotronVLImageProcessor

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<unk>": 1,
            "<img_ctx>": 101,
            "<video_ctx>": 102,
            "<a_start>": 103,
            "<a_ctx>": 104,
            "</a_end>": 105,
        }
        self.unk_token = "<unk>"

    def convert_tokens_to_ids(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = max(self.vocab.values()) + 1
        return self.vocab[token]

    def _tokenize(self, text: str) -> list[str]:
        special = sorted(self.vocab.keys(), key=len, reverse=True)
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            matched = None
            for tok in special:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                tokens.append(matched)
                i += len(matched)
                continue
            j = i + 1
            while j < n and (not text[j].isspace()):
                if any(text.startswith(tok, j) for tok in special):
                    break
                j += 1
            tokens.append(text[i:j])
            i = j
        return tokens

    def __call__(self, text: str, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        token_ids = [self.convert_tokens_to_ids(tok) for tok in self._tokenize(text)]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": token_ids}


class _FakeAudioExtractor:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def audio_token_count(self, audio_len: int) -> int:
        del audio_len
        return 2

    def __call__(self, raw_speech, sampling_rate, return_tensors):
        del sampling_rate, return_tensors
        n = len(raw_speech)
        input_features = torch.zeros((n, 4, 3), dtype=torch.float32)
        attention_mask = torch.ones((n, 4), dtype=torch.long)
        return types.SimpleNamespace(
            input_features=input_features,
            attention_mask=attention_mask,
            audio_num_clips=[1] * n,
        )


def _find_offsets(input_ids: list[int], target_id: int) -> list[tuple[int, int]]:
    offsets = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] != target_id:
            i += 1
            continue
        j = i + 1
        while j < len(input_ids) and input_ids[j] == target_id:
            j += 1
        offsets.append((i, j))
        i = j
    return offsets


def _build_processor_for_audio_tests() -> NanoNemotronVLImageProcessor:
    proc = object.__new__(NanoNemotronVLImageProcessor)
    tokenizer = _FakeTokenizer()

    proc.tokenizer = tokenizer
    proc.mm_tokens = types.SimpleNamespace(
        image_token_id=tokenizer.convert_tokens_to_ids("<img_ctx>"),
        audio_token_id=tokenizer.convert_tokens_to_ids("<a_ctx>"),
    )
    proc.AUDIO_START_TOKEN = "<a_start>"
    proc.AUDIO_CONTEXT_TOKEN = "<a_ctx>"
    proc.AUDIO_END_TOKEN = "</a_end>"
    proc.IMG_CONTEXT_TOKEN = "<img_ctx>"
    proc.IMG_END_TOKEN = "</img>"
    proc.audio_start_token_id = tokenizer.convert_tokens_to_ids(proc.AUDIO_START_TOKEN)
    proc.audio_end_token_id = tokenizer.convert_tokens_to_ids(proc.AUDIO_END_TOKEN)
    proc.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    proc.img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")
    proc.PLACEHOLDER = tokenizer.unk_token
    proc.PLACEHOLDER_ID = tokenizer.convert_tokens_to_ids(proc.PLACEHOLDER)

    proc.dynamic_resolution = False
    proc.video_temporal_patch_size = 1
    proc.num_image_token = 4
    proc.patch_size = 16
    proc.downsample_ratio = 0.5
    proc.min_num_patches = 0
    proc.max_num_patches = 0
    proc.max_model_len = 8192
    proc.norm_mean = (0.5, 0.5, 0.5)
    proc.norm_std = (0.5, 0.5, 0.5)
    proc.video_target_num_patches = 0
    proc.video_maintain_aspect_ratio = True

    proc.audio_extractor = _FakeAudioExtractor()
    proc.evs = types.SimpleNamespace(
        static_size_data_items=lambda **kwargs: (lambda **_k: [], [])
    )
    proc.get_mm_items_offset = _find_offsets
    proc.load_mm_data = lambda **kwargs: types.SimpleNamespace(
        images=[], videos=[], audios=kwargs.get("audio_data") or []
    )
    return proc


class TestNanoNemotronVLProcessorAudio(CustomTestCase):
    def test_render_audio_uses_expected_token_layout(self):
        proc = _build_processor_for_audio_tests()
        rendered = proc.render_audio(num_tokens=3)
        self.assertEqual(rendered, "<a_start><a_ctx><a_ctx><a_ctx></a_end>")

    def test_process_mm_data_replaces_audio_placeholder(self):
        proc = _build_processor_for_audio_tests()
        request = types.SimpleNamespace(video_data=None, use_audio_in_video=False)
        audio = np.zeros(32000, dtype=np.float32)

        out = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=[audio],
                input_text="hello <a_ctx> world",
                request_obj=request,
            )
        )

        self.assertEqual(out.input_ids.count(proc.mm_tokens.audio_token_id), 2)
        self.assertIn(proc.audio_start_token_id, out.input_ids)
        self.assertIn(proc.audio_end_token_id, out.input_ids)
        self.assertEqual(len(out.mm_items), 1)
        item = out.mm_items[0]
        self.assertEqual(item.modality, Modality.AUDIO)
        self.assertEqual(item.model_specific_data["audio_num_clips"], 1)
        self.assertEqual(item.offsets[0][1] - item.offsets[0][0], 2)

    def test_process_mm_data_appends_audio_when_no_placeholder(self):
        proc = _build_processor_for_audio_tests()
        request = types.SimpleNamespace(video_data=None, use_audio_in_video=False)
        audio = np.zeros(32000, dtype=np.float32)

        out = asyncio.run(
            proc.process_mm_data_async(
                image_data=None,
                audio_data=[audio],
                input_text="plain text only",
                request_obj=request,
            )
        )

        self.assertEqual(out.input_ids.count(proc.mm_tokens.audio_token_id), 2)
        self.assertIn(proc.audio_start_token_id, out.input_ids)
        self.assertIn(proc.audio_end_token_id, out.input_ids)
        self.assertEqual(len(out.mm_items), 1)
        self.assertEqual(out.mm_items[0].modality, Modality.AUDIO)

    def test_render_tubelet_single_frame_delegates_to_render_frame(self):
        proc = _build_processor_for_audio_tests()
        rendered = proc.render_tubelet(
            tubelet_index=0,
            frame_indices=[1],
            timestamps=[0.5],
            num_tokens=3,
        )
        self.assertIn("Frame 2 sampled at 0.50 seconds:", rendered)
        self.assertEqual(rendered.count(proc.IMG_CONTEXT_TOKEN), 3)

    def test_render_tubelet_multiple_frames_lists_all_frames(self):
        proc = _build_processor_for_audio_tests()
        rendered = proc.render_tubelet(
            tubelet_index=0,
            frame_indices=[0, 1],
            timestamps=[0.0, 0.5],
            num_tokens=2,
        )
        self.assertIn("frame 1 sampled at 0.00 seconds", rendered)
        self.assertIn("frame 2 sampled at 0.50 seconds", rendered)
        self.assertEqual(rendered.count(proc.IMG_CONTEXT_TOKEN), 2)
        self.assertTrue(rendered.endswith(proc.IMG_END_TOKEN))

    def test_parse_video_uses_sampled_indices_and_computes_timestamps(self):
        class _FakeVideo:
            avg_fps = 4.0

            def get_frames_at(self, frames):
                return np.zeros((len(frames), 2, 2, 3), dtype=np.uint8)

        with patch(
            "sglang.srt.multimodal.processors.nano_nemotron_vl.sample_video_frames",
            return_value=[1, 3, 4],
        ) as mock_sampler:
            frames, timestamps = NanoNemotronVLImageProcessor.parse_video(_FakeVideo())

        mock_sampler.assert_called_once()
        self.assertEqual(frames.shape[0], 3)
        # avg_fps=4.0 -> frame duration 250ms, timestamps = idx*0.25
        self.assertEqual(timestamps, [0.25, 0.75, 1.0])

    def test_process_mm_data_extracts_audio_from_video_when_requested(self):
        proc = _build_processor_for_audio_tests()
        proc.VIDEO_CONTEXT_TOKEN = "<video_ctx>"
        proc.preprocess_image = lambda _image, max_num_tiles=1: torch.zeros(
            (1, 1, 1), dtype=torch.bfloat16
        )
        proc.evs = types.SimpleNamespace(
            static_size_data_items=lambda **kwargs: (lambda **_k: [], [[1]])
        )

        class _VideoWrapper:
            source_bytes = b"fake-video-bytes"
            avg_fps = 1.0

            def __len__(self):
                return 1

            def get_frames_at(self, frames):
                del frames
                return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        proc.load_mm_data = lambda **kwargs: types.SimpleNamespace(
            images=[],
            videos=[_VideoWrapper()],
            audios=[],
        )
        request = types.SimpleNamespace(
            video_data=["dummy-video-input"], use_audio_in_video=True
        )

        with patch(
            "sglang.srt.multimodal.processors.nano_nemotron_vl.extract_audio_from_video_bytes",
            return_value=np.ones(16000, dtype=np.float32),
        ) as mock_extract:
            out = asyncio.run(
                proc.process_mm_data_async(
                    image_data=None,
                    audio_data=[],
                    input_text="prompt with <video_ctx>",
                    request_obj=request,
                )
            )

        mock_extract.assert_called_once_with(b"fake-video-bytes", target_sr=16000)
        audio_items = [item for item in out.mm_items if item.modality == Modality.AUDIO]
        self.assertEqual(len(audio_items), 1)
        self.assertEqual(out.input_ids.count(proc.mm_tokens.audio_token_id), 2)


if __name__ == "__main__":
    unittest.main()
