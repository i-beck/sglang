"""Unit tests for multimodal architecture detection in model_config."""

import unittest

from sglang.srt.configs.model_config import is_multimodal_model, multimodal_model_archs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestModelConfigMultimodalArchs(CustomTestCase):
    def test_nemotron_omni_arch_is_registered_as_multimodal(self):
        self.assertIn("NemotronH_Nano_Omni_Reasoning_V3", multimodal_model_archs)
        self.assertTrue(is_multimodal_model(["NemotronH_Nano_Omni_Reasoning_V3"]))

    def test_non_multimodal_arch_not_detected(self):
        self.assertFalse(is_multimodal_model(["LlamaForCausalLM"]))


if __name__ == "__main__":
    unittest.main()
