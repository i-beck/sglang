"""Unit tests for srt/models/nano_nemotron_vl.py weight routing."""

import unittest

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class _Recorder:
    def __init__(self):
        self.loaded = None

    def load_weights(self, weights):
        self.loaded = list(weights)


class _DummyNemotron:
    def __init__(self, with_sound: bool = True):
        self.language_model = _Recorder()
        self.vision_model = _Recorder()
        self.sound_encoder = _Recorder() if with_sound else None
        self.mlp1 = nn.Sequential(nn.Linear(2, 2, bias=False))


class TestNanoNemotronVLLoadWeights(CustomTestCase):
    def test_load_weights_routes_to_llm_vision_adapter_and_sound(self):
        model = _DummyNemotron(with_sound=True)
        adapter_new = torch.full_like(model.mlp1[0].weight, 7.0)

        NemotronH_Nano_VL_V2.load_weights(
            model,
            [
                ("language_model.layers.0.weight", torch.tensor([1.0])),
                ("vision_model.radio_model.block.weight", torch.tensor([2.0])),
                ("mlp1.0.weight", adapter_new),
                ("sound_encoder.encoder.weight", torch.tensor([3.0])),
            ],
        )

        self.assertEqual(len(model.language_model.loaded), 1)
        self.assertEqual(model.language_model.loaded[0][0], "layers.0.weight")
        self.assertTrue(torch.equal(model.language_model.loaded[0][1], torch.tensor([1.0])))

        self.assertEqual(len(model.vision_model.loaded), 1)
        self.assertEqual(model.vision_model.loaded[0][0], "radio_model.block.weight")
        self.assertTrue(torch.equal(model.vision_model.loaded[0][1], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(model.mlp1[0].weight, adapter_new))
        self.assertEqual(len(model.sound_encoder.loaded), 1)
        self.assertEqual(model.sound_encoder.loaded[0][0], "sound_encoder.encoder.weight")
        self.assertTrue(torch.equal(model.sound_encoder.loaded[0][1], torch.tensor([3.0])))

    def test_sound_weights_ignored_when_sound_encoder_absent(self):
        model = _DummyNemotron(with_sound=False)
        NemotronH_Nano_VL_V2.load_weights(
            model,
            [
                ("sound_encoder.encoder.weight", torch.tensor([3.0])),
                ("language_model.layers.0.weight", torch.tensor([1.0])),
            ],
        )

        self.assertEqual(len(model.language_model.loaded), 1)
        self.assertEqual(model.language_model.loaded[0][0], "layers.0.weight")
        self.assertTrue(torch.equal(model.language_model.loaded[0][1], torch.tensor([1.0])))
        self.assertEqual(model.vision_model.loaded, [])


if __name__ == "__main__":
    unittest.main()
