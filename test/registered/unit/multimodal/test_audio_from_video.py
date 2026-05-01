"""Unit tests for srt/multimodal/audio_from_video.py."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.multimodal.audio_from_video import extract_audio_from_video_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class _FakeAudioFrame:
    def __init__(self, values):
        self._values = values

    def to_ndarray(self):
        return np.asarray(self._values, dtype=np.float32)


class _FakeResampler:
    def __init__(self, *_args, **_kwargs):
        pass

    def resample(self, frame):
        return [frame]


class _FakeStreams:
    def __init__(self, has_audio=True):
        self.audio = [types.SimpleNamespace(rate=16000)] if has_audio else []


class _FakeContainer:
    def __init__(self, has_audio=True, chunks=None):
        self.streams = _FakeStreams(has_audio=has_audio)
        self._chunks = chunks or []
        self.closed = False

    def decode(self, audio=0):
        del audio
        for chunk in self._chunks:
            yield _FakeAudioFrame(chunk)

    def close(self):
        self.closed = True


class TestAudioFromVideo(CustomTestCase):
    def test_returns_none_when_av_missing(self):
        real_import = __import__

        def raising_import(name, *args, **kwargs):
            if name == "av":
                raise ImportError("no av")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raising_import):
            with self.assertLogs(
                "sglang.srt.multimodal.audio_from_video", level="WARNING"
            ) as logs:
                result = extract_audio_from_video_bytes(b"fake")
        self.assertIsNone(result)
        self.assertTrue(any("PyAV (av) is not installed" in line for line in logs.output))

    def test_returns_none_when_container_has_no_audio_stream(self):
        fake_av = types.SimpleNamespace(
            open=lambda _bio: _FakeContainer(has_audio=False),
            audio=types.SimpleNamespace(
                resampler=types.SimpleNamespace(AudioResampler=_FakeResampler)
            ),
        )
        with patch.dict(sys.modules, {"av": fake_av}):
            result = extract_audio_from_video_bytes(b"fake-video")
        self.assertIsNone(result)

    def test_extracts_and_concatenates_audio_chunks(self):
        chunks = [[0.1, -0.1], [0.2], [0.3, 0.4]]
        fake_av = types.SimpleNamespace(
            open=lambda _bio: _FakeContainer(has_audio=True, chunks=chunks),
            audio=types.SimpleNamespace(
                resampler=types.SimpleNamespace(AudioResampler=_FakeResampler)
            ),
        )
        with patch.dict(sys.modules, {"av": fake_av}):
            waveform = extract_audio_from_video_bytes(b"fake-video")

        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.dtype, np.float32)
        np.testing.assert_allclose(waveform, np.array([0.1, -0.1, 0.2, 0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
