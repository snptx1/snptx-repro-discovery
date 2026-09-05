"""SNPTX: experimentation layer for multi-modal biomedical machine learning.

Public package facade for the snptx-core monorepo. Currently exposes the
subset of modules required to reproduce the EXP-01 DeepDDS drug-synergy
result published on the snptx-academic site.

During the v0.1 transition the underlying code still lives at
``src/adapters/``, ``src/preprocessing/``, ``src/models/``. Subpackages
under ``snptx.*`` are thin re-export shims so callers can write
``from snptx.adapters.drugcomb import DrugCombAdapter`` today without
waiting for the physical move (planned for v0.2).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the legacy ``src`` package importable regardless of cwd, so the
# v0.1 re-export shims (which still ``from src.adapters.base import ...``)
# work when this package is installed via ``pip install -e .`` or via the
# GitHub tarball install used by the snptx-repro-deepdds carve-out.
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

__version__ = "0.1.0"

from snptx.repro.determinism import seed_worker, set_seed, snapshot_env  # noqa: E402

__all__ = ["__version__", "set_seed", "seed_worker", "snapshot_env"]
