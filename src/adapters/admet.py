"""ADMET property prediction adapter (TDC-backed).

Wraps the Therapeutics Data Commons ADMET single-prediction tasks with
optional compound filtering via :mod:`src.preprocessing.compound_filters`.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pandas as pd

from src.adapters.base import TabularAdapter
from src.adapters.registry import AdapterRegistry

logger = logging.getLogger(__name__)


@AdapterRegistry.register("admet")
class ADMETAdapter(TabularAdapter):
    """Adapter for ADMET property prediction benchmarks.

    Each endpoint corresponds to a TDC dataset name (see :attr:`ADMET_TDC_MAP`).
    The returned DataFrame has at least ``Drug`` (SMILES) and the canonical
    target column ``property_value`` (re-named from TDC's ``Y``).
    """

    ADMET_TDC_MAP: ClassVar[dict[str, str]] = {
        "caco2": "Caco2_Wang",
        "hia": "HIA_Hou",
        "pgp": "Pgp_Broccatelli",
        "bioavailability": "Bioavailability_Ma",
        "lipophilicity": "Lipophilicity_AstraZeneca",
        "solubility": "Solubility_AqSolDB",
        "bbb": "BBB_Martins",
        "ppbr": "PPBR_AZ",
        "clearance": "Clearance_Hepatocyte_AZ",
        "half_life": "Half_Life_Obach",
        "herg": "hERG",
        "ames": "AMES",
        "ld50": "LD50_Zhu",
    }
    TOX_DATASETS: ClassVar[set[str]] = {"hERG", "AMES", "LD50_Zhu"}

    @property
    def name(self) -> str:
        return "admet"

    @property
    def supported_endpoints(self) -> list[str]:
        return list(self.ADMET_TDC_MAP.keys())

    # ------------------------------------------------------------------
    def build(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        self.validate_endpoint(endpoint)
        dataset_name = self.ADMET_TDC_MAP[endpoint]
        apply_filters: bool = kwargs.get("apply_filters", False)
        split: str = kwargs.get("split", "all")

        df = self._load_tdc(dataset_name, split=split)

        # Standardize columns
        if "Drug" not in df.columns:
            for cand in ("smiles", "SMILES"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "Drug"})
                    break
        if "Y" in df.columns:
            df = df.rename(columns={"Y": "property_value"})

        if apply_filters:
            from src.preprocessing.compound_filters import filter_dataframe

            df = filter_dataframe(df, smiles_col="Drug", drop_failures=True)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def _load_tdc(self, dataset_name: str, split: str = "all") -> pd.DataFrame:
        from tdc.single_pred import ADME, Tox  # type: ignore[import-not-found]

        cache = str(self.raw_dir)
        loader_cls = Tox if dataset_name in self.TOX_DATASETS else ADME
        data = loader_cls(name=dataset_name, path=cache)
        if split == "all":
            return pd.DataFrame(data.get_data())
        splits = data.get_split(method="scaffold")
        return pd.DataFrame(splits[split])

    def _target_column(self, endpoint: str) -> str:
        # All ADMET endpoints share the canonical column ``property_value``
        return "property_value"
