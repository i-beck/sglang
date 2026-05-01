"""Unit tests for srt/configs/nano_nemotron_vl.py."""

import unittest

from transformers import PretrainedConfig

from sglang.srt.configs.nano_nemotron_vl import (
    NemotronH_Nano_Omni_Reasoning_V3_Config,
    NemotronH_Nano_VL_V2_Config,
    float_triplet,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestNanoNemotronVLConfig(CustomTestCase):
    def test_float_triplet_accepts_three_floats(self):
        self.assertEqual(float_triplet((0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))

    def test_float_triplet_rejects_non_float(self):
        with self.assertRaises(AssertionError):
            float_triplet((1.0, 2, 3.0))

    def test_sound_config_dict_is_converted_to_pretrained_config(self):
        cfg = NemotronH_Nano_VL_V2_Config(
            llm_config={"hidden_size": 64, "num_hidden_layers": 1},
            vision_config={"args": {"model": "vit_huge_patch16_224"}},
            sound_config={"model_type": "parakeet-encoder", "sampling_rate": 16000},
        )
        self.assertIsInstance(cfg.sound_config, PretrainedConfig)
        self.assertEqual(cfg.sound_config.sampling_rate, 16000)

    def test_dynamic_and_temporal_fields_propagate_from_vision_config(self):
        cfg = NemotronH_Nano_VL_V2_Config(
            llm_config={"hidden_size": 64, "num_hidden_layers": 1},
            vision_config={
                "image_size": [768],
                "patch_size": [16],
                "args": {"model": "vit_huge_patch16_224", "register_multiple": 4},
                "preferred_resolution": [448],
                "min_num_patches": 8,
                "max_num_patches": 64,
                "video_temporal_patch_size": 2,
                "separate_video_embedder": False,
                "video_target_num_patches": 32,
                "video_maintain_aspect_ratio": False,
            },
        )

        self.assertTrue(cfg.dynamic_resolution)
        self.assertEqual(cfg.min_num_patches, 8)
        self.assertEqual(cfg.max_num_patches, 64)
        self.assertEqual(cfg.video_temporal_patch_size, 2)
        self.assertFalse(cfg.separate_video_embedder)
        self.assertEqual(cfg.video_target_num_patches, 32)
        self.assertFalse(cfg.video_maintain_aspect_ratio)

        radio_cfg = cfg.create_radio_config()
        self.assertEqual(radio_cfg.model_name, "vit_huge_patch16_224")
        self.assertEqual(radio_cfg.reg_tokens, 4)
        self.assertEqual(radio_cfg.image_size, 448)
        self.assertEqual(radio_cfg.min_num_patches, 8)
        self.assertEqual(radio_cfg.max_num_patches, 64)
        self.assertEqual(radio_cfg.video_temporal_patch_size, 2)
        self.assertFalse(radio_cfg.separate_video_embedder)
        self.assertEqual(radio_cfg.video_target_num_patches, 32)
        self.assertFalse(radio_cfg.video_maintain_aspect_ratio)

    def test_omni_config_inherits_base_behavior(self):
        cfg = NemotronH_Nano_Omni_Reasoning_V3_Config(
            llm_config={"hidden_size": 64, "num_hidden_layers": 1},
            vision_config={"args": {"model": "vit_huge_patch16_224"}},
        )
        self.assertEqual(cfg.model_type, "NemotronH_Nano_Omni_Reasoning_V3")
        self.assertIsInstance(cfg, NemotronH_Nano_VL_V2_Config)


if __name__ == "__main__":
    unittest.main()
