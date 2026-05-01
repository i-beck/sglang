"""Unit tests for srt/models/parakeet.py."""

import unittest

import numpy as np
from transformers import PretrainedConfig

from sglang.srt.models.parakeet import ParakeetExtractor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


def _make_hf_config(**overrides) -> PretrainedConfig:
    base = dict(
        num_mel_bins=80,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        hop_length=160,
    )
    base.update(overrides)
    return PretrainedConfig(**base)


class TestParakeetExtractor(CustomTestCase):
    def setUp(self):
        self.extractor = ParakeetExtractor(_make_hf_config())

    def test_clip_sizes_include_tail_clip(self):
        clip_target = int(round(self.extractor.config.clip_duration_s * 16000))
        tail = int(round(self.extractor.config.clip_min_duration_s * 16000))
        audio_len = clip_target + tail // 2
        self.assertEqual(self.extractor._clip_sizes(audio_len), [clip_target, tail])

    def test_split_audio_into_clips_pads_to_min_tail(self):
        clip_target = int(round(self.extractor.config.clip_duration_s * 16000))
        short_tail = 100
        audio = np.zeros(clip_target + short_tail, dtype=np.float32)

        clips = self.extractor.split_audio_into_clips(audio)

        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0].shape[0], clip_target)
        self.assertGreaterEqual(clips[1].shape[0], int(0.1 * 16000))

    def test_audio_token_count_matches_clipwise_subsampling_sum(self):
        clip_target = int(round(self.extractor.config.clip_duration_s * 16000))
        audio_len = clip_target * 2 + 32000

        expected = 0
        for clip_size in self.extractor._clip_sizes(audio_len):
            frames = clip_size // self.extractor.hop_length
            expected += self.extractor._subsampling_output_length(frames)

        self.assertEqual(self.extractor.audio_token_count(audio_len), expected)

    def test_audio_length_inverse_formula(self):
        raw_cfg = _make_hf_config(subsampling_factor=4, hop_length=200)
        self.assertEqual(ParakeetExtractor.audio_length(raw_cfg, audio_tokens=7), 5600)


if __name__ == "__main__":
    unittest.main()
