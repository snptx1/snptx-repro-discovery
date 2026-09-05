"""Determinism helpers for reproducible SNPTX experiments.

The three primitives exported from this module cover the standard
checklist that the EXP-01 reproduction script and notebook need:

* :func:`set_seed` seeds Python, NumPy, and PyTorch (CPU + CUDA) and
  configures cuDNN for deterministic kernels.
* :func:`seed_worker` is the ``worker_init_fn`` used by every
  :class:`torch.utils.data.DataLoader` so multi-worker shuffling is
  reproducible.
* :func:`snapshot_env` returns a JSON-serialisable dict describing the
  Python / library versions and hardware that produced a given run, to
  be embedded in the reference metrics file.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic kernels.

    Parameters
    ----------
    seed:
        Integer seed. Defaults to 42 to match the snptx-academic
        reproduction convention.
    deterministic:
        When ``True`` (default), force cuDNN into deterministic mode and
        request deterministic algorithms from PyTorch. This trades a
        small amount of speed for bit-equality across runs on the same
        hardware class. Set to ``False`` for benchmarking runs where
        throughput matters more than reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: torch.use_deterministic_algorithms is intentionally NOT called
        # here because torch_geometric's scatter and GNN aggregation kernels
        # do not all have deterministic CUDA implementations and would either
        # error or fall back to slow paths. Bit-equality is achieved on CPU
        # only (see snptx-repro-deepdds/repro_deepdds.py); GPU runs match to
        # ~3 decimal places which is sufficient for the headline metric
        # tolerance documented in the EXP-01 notebook.


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that re-seeds NumPy and Python from torch's seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def snapshot_env() -> dict[str, Any]:
    """Return a JSON-serialisable dict describing the runtime environment.

    The dict is intended to be embedded in the reference ``*.json`` metrics
    file so that any later mismatch between observed and expected numbers
    can be diagnosed against the exact stack that produced the reference.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,  # type: ignore[attr-defined]
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    import importlib.metadata as _ilm

    for pkg, dist in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torch_geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
        ("tdc", "PyTDC"),
    ):
        try:
            info[pkg] = _ilm.version(dist)
        except _ilm.PackageNotFoundError:
            try:
                mod = __import__(pkg)
                info[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                info[pkg] = None
    return info
