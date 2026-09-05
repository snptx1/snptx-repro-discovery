"""Drug-likeness compound filters: Lipinski, PAINS, REOS, Glaxo.

All filters are pure functions that operate on RDKit ``Mol`` objects (or
SMILES strings) and return either ``(passes, info)`` tuples or a fully
populated :class:`FilterResult` dataclass.

References:
    Lipinski et al. "Experimental and computational approaches to
        estimate solubility and permeability in drug discovery and
        development settings", Adv. Drug Deliv. Rev., 1997.
    Baell & Holloway. "New substructure filters for removal of pan
        assay interference compounds (PAINS)", J. Med. Chem., 2010.
    Walters et al. "Virtual screening — an overview" (REOS), Drug
        Discov. Today, 1998.
    Hann et al. (Glaxo) "Strategic pooling of compounds for HTS",
        J. Chem. Inf. Comput. Sci., 1999.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rdkit import Chem  # type: ignore[import-untyped]
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Outcome of running all compound filters on a single molecule."""

    smiles: str
    passes_lipinski: bool = False
    passes_pains: bool = False
    passes_reos: bool = False
    passes_glaxo: bool = False
    lipinski_violations: int = 0
    flags: list[str] = field(default_factory=list)
    valid: bool = True

    @property
    def passes_all(self) -> bool:
        return (
            self.valid
            and self.passes_lipinski
            and self.passes_pains
            and self.passes_reos
            and self.passes_glaxo
        )


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------
def lipinski_filter(mol: Chem.Mol) -> tuple[bool, int]:
    """Lipinski's Rule of Five — passes if violations ≤ 1."""
    mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    hbd = Lipinski.NumHDonors(mol)  # type: ignore[attr-defined]
    hba = Lipinski.NumHAcceptors(mol)  # type: ignore[attr-defined]
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return violations <= 1, violations


def pains_filter(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """Screen against PAINS substructures (RDKit built-in catalog)."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    entry = catalog.GetFirstMatch(mol)
    if entry is None:
        return True, []
    return False, [f"PAINS:{entry.GetDescription()}"]


def reos_filter(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """Rapid Elimination Of Swill (Walters et al. 1998)."""
    mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    hbd = Lipinski.NumHDonors(mol)  # type: ignore[attr-defined]
    hba = Lipinski.NumHAcceptors(mol)  # type: ignore[attr-defined]
    rotb = Lipinski.NumRotatableBonds(mol)  # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    rings = Lipinski.RingCount(mol)  # type: ignore[attr-defined]
    heavy = mol.GetNumHeavyAtoms()
    flags: list[str] = []
    if not 200 <= mw <= 500:
        flags.append(f"REOS:MW={mw:.1f}")
    if not -5 <= logp <= 5:
        flags.append(f"REOS:logP={logp:.2f}")
    if hbd > 5:
        flags.append(f"REOS:HBD={hbd}")
    if hba > 10:
        flags.append(f"REOS:HBA={hba}")
    if rotb > 8:
        flags.append(f"REOS:RotB={rotb}")
    if tpsa > 150:
        flags.append(f"REOS:TPSA={tpsa:.1f}")
    if not 0 <= rings <= 6:
        flags.append(f"REOS:rings={rings}")
    if not 15 <= heavy <= 50:
        flags.append(f"REOS:heavy={heavy}")
    return len(flags) == 0, flags


def glaxo_filter(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """Glaxo (GSK) lead-likeness rules — stricter MW/logP than REOS."""
    mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    rotb = Lipinski.NumRotatableBonds(mol)  # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    flags: list[str] = []
    if not 100 <= mw <= 450:
        flags.append(f"GSK:MW={mw:.1f}")
    if not -2 <= logp <= 4:
        flags.append(f"GSK:logP={logp:.2f}")
    if rotb > 8:
        flags.append(f"GSK:RotB={rotb}")
    if tpsa > 140:
        flags.append(f"GSK:TPSA={tpsa:.1f}")
    return len(flags) == 0, flags


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def filter_compound(smiles: str) -> FilterResult:
    """Run every filter on a single SMILES string.

    Invalid SMILES return a :class:`FilterResult` with ``valid=False`` and
    every individual ``passes_*`` flag set to ``False``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return FilterResult(smiles=smiles, valid=False, flags=["INVALID_SMILES"])

    pass_lip, n_viol = lipinski_filter(mol)
    pass_pains, pains_flags = pains_filter(mol)
    pass_reos, reos_flags = reos_filter(mol)
    pass_glaxo, glaxo_flags = glaxo_filter(mol)

    return FilterResult(
        smiles=smiles,
        passes_lipinski=pass_lip,
        passes_pains=pass_pains,
        passes_reos=pass_reos,
        passes_glaxo=pass_glaxo,
        lipinski_violations=n_viol,
        flags=pains_flags + reos_flags + glaxo_flags,
    )


def filter_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "Drug",
    drop_failures: bool = False,
    require: tuple[str, ...] = ("lipinski", "pains"),
) -> pd.DataFrame:
    """Annotate a DataFrame with per-compound filter outcomes.

    Parameters
    ----------
    df:
        Input DataFrame containing a SMILES column.
    smiles_col:
        Name of the SMILES column (default ``"Drug"`` per TDC convention).
    drop_failures:
        If ``True``, remove rows that do not pass every filter listed in
        ``require``.
    require:
        Subset of {"lipinski", "pains", "reos", "glaxo"} that compounds
        must satisfy when ``drop_failures=True``.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"smiles_col '{smiles_col}' not in DataFrame columns: {list(df.columns)}")

    records: list[dict[str, Any]] = []
    for smi in df[smiles_col].astype(str):
        r = filter_compound(smi)
        records.append(
            {
                "passes_lipinski": r.passes_lipinski,
                "passes_pains": r.passes_pains,
                "passes_reos": r.passes_reos,
                "passes_glaxo": r.passes_glaxo,
                "lipinski_violations": r.lipinski_violations,
                "filter_flags": ";".join(r.flags),
                "valid_smiles": r.valid,
            }
        )
    annotated = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    if drop_failures:
        mask = annotated["valid_smiles"].astype(bool)
        for rule in require:
            col = f"passes_{rule}"
            if col in annotated.columns:
                mask &= annotated[col].astype(bool)
        n_before = len(annotated)
        annotated = pd.DataFrame(annotated[mask]).reset_index(drop=True)
        logger.info("filter_dataframe: kept %d / %d rows", len(annotated), n_before)

    return annotated
