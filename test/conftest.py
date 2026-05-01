import os
import sys
from pathlib import Path

# Ensure local package imports work when pytest rootdir is test/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# Make pytest runs robust on CPU/non-CUDA test environments by avoiding
# optional CUDA-only imports in transitive module import paths.
os.environ.setdefault("SGLANG_TEST_SKIP_OPTIONAL_GPU_IMPORTS", "1")
