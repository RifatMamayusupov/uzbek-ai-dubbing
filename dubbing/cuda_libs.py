"""Make pip-installed cuDNN/cuBLAS visible to CTranslate2 (faster-whisper).

CTranslate2 ``dlopen``s ``libcudnn_ops.so.9`` by name and does not search the
``nvidia-*-cu12`` wheel directories, which crashes the process with
"Unable to load any of {libcudnn_ops.so...}". Loading those libraries into the
process first (RTLD_GLOBAL) lets the later ``dlopen`` find them.
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_LIBS = {
    "nvidia.cublas": ["libcublasLt.so.12", "libcublas.so.12"],
    "nvidia.cudnn": ["libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9",
                     "libcudnn_cnn.so.9", "libcudnn_adv.so.9"],
}
_done = False


def preload() -> None:
    global _done
    if _done or not sys.platform.startswith("linux"):
        return
    _done = True
    for package, names in _LIBS.items():
        spec = importlib.util.find_spec(package)
        if not spec or not spec.submodule_search_locations:
            continue
        lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        for name in names:
            path = lib_dir / name
            if path.exists():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    log.debug("Could not preload %s: %s", path, exc)
