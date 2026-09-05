"""Drug combination synergy adapter for the DeepDDS pipeline.

Loads drug-pair synergy tables (DrugCombDB / NCI-ALMANAC / TDC's
``DrugComb`` benchmark) and converts SMILES into PyG molecular graphs.

Reference:
    Liu et al. "DrugCombDB: a comprehensive database of drug
    combinations toward the discovery of combinatorial therapy",
    Nucleic Acids Res. 2020.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data

from src.adapters.base import GraphAdapter
from src.adapters.registry import AdapterRegistry

logger = logging.getLogger(__name__)

# Atom feature dimension produced by ``smiles_to_graph``.
ATOM_FEATURE_DIM: int = 9
EDGE_FEATURE_DIM: int = 3


@AdapterRegistry.register("drugcomb")
class DrugCombAdapter(GraphAdapter):
    """Drug-pair synergy adapter."""

    @property
    def name(self) -> str:
        return "drugcomb"

    @property
    def supported_endpoints(self) -> list[str]:
        return ["drug_synergy", "drug_pair_features"]

    # ------------------------------------------------------------------
    def build(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        self.validate_endpoint(endpoint)
        return self._load_synergy(**kwargs)

    # ------------------------------------------------------------------
    def _load_synergy(
        self,
        source: str = "tdc",
        synergy_threshold: float = 10.0,
        csv_path: str | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        """Load a drug-pair synergy table.

        Parameters
        ----------
        source:
            ``"tdc"`` to download via TDC's ``DrugSyn`` group, ``"csv"`` to
            read a local CSV at ``csv_path``.
        synergy_threshold:
            Loewe score above which a pair is labelled synergistic.
        csv_path:
            Path to a local CSV when ``source="csv"``. Must contain
            columns: ``drug1_smiles``, ``drug2_smiles``, ``synergy_score``,
            and optionally ``cell_line``.
        """
        if source == "csv":
            if csv_path is None:
                raise ValueError("csv_path is required when source='csv'")
            df = pd.read_csv(csv_path)
        else:
            df = self._load_via_tdc()

        df = self._standardize_columns(df)
        if "synergy_score" not in df.columns:
            raise ValueError(
                f"Expected 'synergy_score' column in synergy table. Got: {list(df.columns)}"
            )

        df = df.dropna(subset=["drug1_smiles", "drug2_smiles", "synergy_score"]).reset_index(drop=True)
        df["synergy_label"] = (df["synergy_score"] > synergy_threshold).astype(int)
        return df

    def _load_via_tdc(self) -> pd.DataFrame:
        from tdc.multi_pred import DrugSyn  # type: ignore[import-not-found]

        cache = str(self.raw_dir)
        data = DrugSyn(name="DrugComb", path=cache)
        return pd.DataFrame(data.get_data())

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {
            "Drug1": "drug1_smiles",
            "Drug2": "drug2_smiles",
            "Drug1_SMILES": "drug1_smiles",
            "Drug2_SMILES": "drug2_smiles",
            "smiles_1": "drug1_smiles",
            "smiles_2": "drug2_smiles",
            "Y": "synergy_score",
            "Synergy_Loewe": "synergy_score",
            "Cell_Line": "cell_line",
            "Cell_Line_ID": "cell_line",
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})
        return df

    # ------------------------------------------------------------------
    # SMILES → molecular graph
    # ------------------------------------------------------------------
    def smiles_to_graph(self, smiles: str) -> Data | None:
        """Convert a SMILES string into a PyG :class:`Data` object.

        Atom features (9 dims): atomic number, degree, formal charge,
        explicit H count, aromatic flag, in-ring flag, one-hot
        hybridization (SP / SP2 / SP3).

        Bond features (3 dims): bond order, conjugated flag, in-ring flag.
        Edges are added in both directions to keep the graph undirected.
        """
        from rdkit import Chem  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features: list[list[float]] = []
        for atom in mol.GetAtoms():  # type: ignore[call-arg]
            hyb = atom.GetHybridization()
            atom_features.append(
                [
                    float(atom.GetAtomicNum()),
                    float(atom.GetDegree()),
                    float(atom.GetFormalCharge()),
                    float(atom.GetNumExplicitHs()),
                    float(int(atom.GetIsAromatic())),
                    float(int(atom.IsInRing())),
                    float(int(hyb == Chem.rdchem.HybridizationType.SP)),
                    float(int(hyb == Chem.rdchem.HybridizationType.SP2)),
                    float(int(hyb == Chem.rdchem.HybridizationType.SP3)),
                ]
            )
        x = torch.tensor(atom_features, dtype=torch.float32)

        edge_index_list: list[list[int]] = []
        edge_attr_list: list[list[float]] = []
        for bond in mol.GetBonds():  # type: ignore[call-arg]
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = [
                float(bond.GetBondTypeAsDouble()),
                float(int(bond.GetIsConjugated())),
                float(int(bond.IsInRing())),
            ]
            edge_index_list.extend([[i, j], [j, i]])
            edge_attr_list.extend([bf, bf])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # ------------------------------------------------------------------
    def build_pair_dataset(
        self,
        df: pd.DataFrame,
        label_col: str = "synergy_label",
    ) -> list[tuple[Data, Data, torch.Tensor]]:
        """Convert a drug-pair DataFrame to ``(graph_a, graph_b, label)``."""
        if not {"drug1_smiles", "drug2_smiles", label_col}.issubset(df.columns):
            raise KeyError(
                f"Expected columns drug1_smiles, drug2_smiles, {label_col}. "
                f"Got: {list(df.columns)}"
            )

        cache: dict[str, Data | None] = {}
        pairs: list[tuple[Data, Data, torch.Tensor]] = []
        for _, row in df.iterrows():
            s1, s2 = str(row["drug1_smiles"]), str(row["drug2_smiles"])
            if s1 not in cache:
                cache[s1] = self.smiles_to_graph(s1)
            if s2 not in cache:
                cache[s2] = self.smiles_to_graph(s2)
            g1, g2 = cache[s1], cache[s2]
            if g1 is None or g2 is None:
                continue
            label = torch.tensor([float(row[label_col])], dtype=torch.float32)
            pairs.append((g1.clone(), g2.clone(), label))

        n_total = max(len(df), 1)
        logger.info(
            "Built %d graph pairs from %d rows (%.1f%% valid)",
            len(pairs),
            len(df),
            100 * len(pairs) / n_total,
        )
        return pairs
