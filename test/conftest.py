import os

# Make pytest runs robust on CPU/non-CUDA test environments by avoiding
# optional CUDA-only imports in transitive module import paths.
os.environ.setdefault("SGLANG_TEST_SKIP_OPTIONAL_GPU_IMPORTS", "1")
