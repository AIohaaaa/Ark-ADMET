"""Fingerprint block shared by every endpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import DataStructs, rdReducedGraphs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def _counts(fingerprint) -> np.ndarray:
    array = np.zeros((0,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def fingerprints(smiles: pd.Series, descriptor_names: list[str]) -> np.ndarray:
    """Morgan counts (r=2, 1024), Avalon counts (1024), ErG (315), RDKit descriptors (200)."""
    RDLogger.DisableLog("rdApp.*")
    molecules = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
    valid_idx = [i for i, mol in enumerate(molecules) if mol is not None]
    if not valid_idx:
        raise RuntimeError("no parseable SMILES")
    valid = pd.Series([molecules[i] for i in valid_idx])
    calculator = MolecularDescriptorCalculator(descriptor_names)
    morgan = np.stack(valid.apply(lambda mol: _counts(GetHashedMorganFingerprint(mol, nBits=1024, radius=2))))
    avalon = np.stack(valid.apply(lambda mol: _counts(GetAvalonCountFP(mol, nBits=1024))))
    erg = np.stack(valid.apply(rdReducedGraphs.GetErGFingerprint).to_numpy())
    desc = np.vstack(valid.apply(lambda mol: np.asarray(calculator.CalcDescriptors(mol), dtype=float)))
    parsed = np.concatenate([morgan, avalon, erg, desc], axis=1)
    out = np.full((len(molecules), parsed.shape[1]), np.nan, dtype=float)
    out[np.asarray(valid_idx)] = parsed
    if parsed.shape[1] != 2563:
        raise RuntimeError(f"fingerprint width {parsed.shape[1]} != 2563")
    return out


def row_index(pool: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    """Match split rows back to train_val, including duplicate (SMILES, label) pairs."""
    buckets: dict[tuple[str, float], list[int]] = {}
    for i, (one, y) in enumerate(zip(pool["Drug"].astype(str), pool["Y"].to_numpy(float))):
        buckets.setdefault((one, float(y)), []).append(i)
    cursor = {key: 0 for key in buckets}
    chosen = []
    for one, y in zip(part["Drug"].astype(str), part["Y"].to_numpy(float)):
        key = (one, float(y))
        position = cursor[key]
        chosen.append(buckets[key][position])
        cursor[key] = position + 1
    return np.asarray(chosen, dtype=int)


def assemble(fp: np.ndarray, padel: np.ndarray | None, padel_index: list[int], gin: np.ndarray | None, minimol: np.ndarray | None, concat_minimol: bool) -> np.ndarray:
    blocks = [fp]
    if padel_index:
        if padel is None:
            raise RuntimeError("this endpoint keeps PaDEL columns but none were computed")
        blocks.append(np.asarray(padel, dtype=float)[:, padel_index])
    if gin is not None:
        blocks.append(np.asarray(gin, dtype=float))
    if concat_minimol:
        if minimol is None:
            raise RuntimeError("this endpoint concatenates MiniMol but no embedding was computed")
        blocks.append(np.asarray(minimol, dtype=float))
    return np.ascontiguousarray(np.hstack(blocks))
