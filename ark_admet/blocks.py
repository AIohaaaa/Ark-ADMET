"""Optional PaDEL, GIN, and MiniMol blocks. Each is cached as a .npy aligned to the SMILES list."""
from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def _load_or_build(path: Path, smiles: list[str], builder) -> np.ndarray:
    if path.is_file():
        saved = np.load(path)
        if len(saved) == len(smiles):
            return saved
    array = builder(smiles)
    if len(array) != len(smiles):
        raise RuntimeError(f"{path.name} returned {len(array)} rows for {len(smiles)} SMILES")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return array


def padel_matrix(smiles: list[str], dest: Path) -> np.ndarray:
    return _load_or_build(dest, smiles, _compute_padel)


def gin_matrix(smiles: list[str], dest: Path, device: str) -> np.ndarray:
    return _load_or_build(dest, smiles, lambda rows: _compute_gin(rows, device))


def minimol_matrix(smiles: list[str], dest: Path) -> np.ndarray:
    return _load_or_build(dest, smiles, _compute_minimol)


def _padel_jar() -> tuple[str, Path, Path]:
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java is required to run PaDEL. Install a JRE and put java on PATH.")
    try:
        import padelpy
    except ImportError as exc:
        raise RuntimeError("pip install padelpy  (see requirements-optional.txt)") from exc
    root = Path(padelpy.__file__).resolve().parent / "PaDEL-Descriptor"
    jar = root / "PaDEL-Descriptor.jar"
    types_src = root / "descriptors.xml"
    if not jar.is_file() or not types_src.is_file():
        raise RuntimeError(f"PaDEL jar not found under {root}")
    return java, jar, types_src


def _write_3d_types(src: Path, dest: Path) -> None:
    tree = ET.parse(src)
    for group in tree.getroot():
        keep = group.get("name") == "3D"
        for desc in group:
            desc.set("value", "true" if keep else "false")
    tree.write(dest, encoding="unicode")


def _conformer_sdf(smiles: list[str], path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    writer = Chem.SDWriter(str(path))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol.SetProp("_Name", f"mol{i:05d}")
        writer.write(mol)
    writer.close()


def _run_padel(java: str, jar: Path, sdf: Path, csv_path: Path, types: Path, timeout: int) -> pd.DataFrame | None:
    if csv_path.exists():
        csv_path.unlink()
    cmd = [
        java, "-Djava.awt.headless=true", "-jar", str(jar),
        "-3d", "-retain3d", "-retainorder",
        "-descriptortypes", str(types),
        "-dir", str(sdf),
        "-file", str(csv_path),
        "-threads", "4",
        "-maxruntime", "60000",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not csv_path.is_file():
        return None
    frame = pd.read_csv(csv_path)
    if frame.empty or "Name" not in frame.columns:
        return None
    return frame


def _compute_padel(smiles: list[str]) -> np.ndarray:
    java, jar, types_src = _padel_jar()
    work = Path("cache") / "padel_work"
    work.mkdir(parents=True, exist_ok=True)
    sdf_path = work / "conformers.sdf"
    types = work / "3d.xml"
    _conformer_sdf(smiles, sdf_path)
    _write_3d_types(types_src, types)
    from rdkit import Chem

    mols = [mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False) if mol is not None]
    pieces: list[pd.DataFrame] = []
    batch_dir = work / "batches"
    batch_dir.mkdir(exist_ok=True)
    for start in range(0, len(mols), 40):
        batch = mols[start : start + 40]
        batch_sdf = batch_dir / f"batch_{start:05d}.sdf"
        writer = Chem.SDWriter(str(batch_sdf))
        for mol in batch:
            writer.write(mol)
        writer.close()
        frame = _run_padel(java, jar, batch_sdf, batch_sdf.with_suffix(".csv"), types, timeout=90)
        if frame is None:
            for mol in batch:
                one_sdf = batch_dir / f"{mol.GetProp('_Name')}.sdf"
                one_writer = Chem.SDWriter(str(one_sdf))
                one_writer.write(mol)
                one_writer.close()
                one = _run_padel(java, jar, one_sdf, one_sdf.with_suffix(".csv"), types, timeout=25)
                if one is not None:
                    pieces.append(one)
            continue
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("PaDEL returned no 3D descriptors")
    raw = pd.concat(pieces, ignore_index=True)
    expected = json.loads((Path(__file__).resolve().parents[1] / "configs" / "endpoints.json").read_text())["padel_columns"]
    missing = [name for name in expected if name not in raw.columns]
    if missing:
        raise RuntimeError(f"PaDEL output is missing {missing[:5]}")
    indexed = {}
    for name, row in zip(raw["Name"], raw[expected].to_numpy(float)):
        indexed[int(str(name).removeprefix("mol"))] = row
    out = np.full((len(smiles), len(expected)), np.nan, dtype=float)
    for i, row in indexed.items():
        out[i] = row
    return out


def _compute_gin(smiles: list[str], device: str) -> np.ndarray:
    import dgllife
    import torch
    from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph
    from rdkit import Chem, RDLogger
    from torch.utils.data import DataLoader

    RDLogger.DisableLog("rdApp.*")
    model = dgllife.model.load_pretrained("gin_supervised_masking")
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device).eval()
    node_featurizer = PretrainAtomFeaturizer()
    edge_featurizer = PretrainBondFeaturizer()
    graphs = []
    success = []
    for one in smiles:
        mol = Chem.MolFromSmiles(one) if isinstance(one, str) else None
        if mol is None:
            success.append(False)
            continue
        try:
            graphs.append(mol_to_bigraph(
                mol,
                add_self_loop=True,
                node_featurizer=node_featurizer,
                edge_featurizer=edge_featurizer,
                canonical_atom_order=False,
            ))
            success.append(True)
        except Exception:
            success.append(False)
    out = np.full((len(smiles), 300), np.nan, dtype=np.float64)
    if not graphs:
        return out
    import dgl
    from dgl.nn.pytorch.glob import AvgPooling

    pool = AvgPooling().to(torch_device)
    loader = DataLoader(graphs, batch_size=128, collate_fn=dgl.batch, shuffle=False)
    pieces = []
    for bg in loader:
        bg = bg.to(torch_device)
        nfeats = [bg.ndata.pop("atomic_number"), bg.ndata.pop("chirality_type")]
        efeats = [bg.edata.pop("bond_type"), bg.edata.pop("bond_direction_type")]
        with torch.no_grad():
            node = model(bg, nfeats, efeats)
            pieces.append(pool(bg, node).detach().float().cpu())
    packed = torch.cat(pieces, dim=0).numpy()
    cursor = 0
    for i, ok in enumerate(success):
        if ok:
            out[i] = packed[cursor]
            cursor += 1
    return out


def _compute_minimol(smiles: list[str]) -> np.ndarray:
    import torch
    from minimol import Minimol

    model = Minimol()
    out = np.full((len(smiles), 512), np.nan, dtype=np.float32)
    batch = 64
    for start in range(0, len(smiles), batch):
        chunk = smiles[start : start + batch]
        try:
            with torch.no_grad():
                rows = model(chunk)
            if len(rows) != len(chunk):
                raise RuntimeError("batch length mismatch")
            for offset, vector in enumerate(rows):
                array = vector.detach().float().cpu().numpy()
                if array.ndim == 1 and array.shape[0] == 512 and np.isfinite(array).all():
                    out[start + offset] = array
        except Exception:
            for offset, one in enumerate(chunk):
                try:
                    with torch.no_grad():
                        rows = model([one])
                    array = rows[0].detach().float().cpu().numpy()
                    if array.ndim == 1 and array.shape[0] == 512 and np.isfinite(array).all():
                        out[start + offset] = array
                except Exception:
                    continue
    return out
