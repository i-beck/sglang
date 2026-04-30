import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_sm,
    is_blackwell_supported,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()
_test_skip_optional_gpu_imports = get_bool_env_var(
    "SGLANG_TEST_SKIP_OPTIONAL_GPU_IMPORTS"
)


def _compute_enable_deep_gemm():
    if _test_skip_optional_gpu_imports and not (_is_cuda or _is_musa):
        return False

    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
