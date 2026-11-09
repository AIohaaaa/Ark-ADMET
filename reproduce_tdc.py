"""Train the frozen Ark-ADMET configuration and score it with TDC evaluate_many."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

from ark_admet.blocks import gin_matrix, minimol_matrix, padel_matrix
from ark_admet.features import assemble, fingerprints, row_index
from ark_admet.model import fit_catboost, fit_head, mix

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "endpoints.json").read_text())
SEEDS = tuple(CONFIG["seeds"])


def needs(cfg: dict) -> tuple[bool, bool, bool]:
    padel = int(cfg["n_3d"]) > 0
    gin = bool(cfg["gin"])
    minimol = bool(cfg["minimol_concat"]) or float(cfg["blend_weight"]) > 0
    return padel, gin, minimol


def block_matrix(name: str, split: str, frame: pd.DataFrame, cfg: dict, cache: Path, device: str):
    smiles = frame["Drug"].astype(str).tolist()
    folder = cache / name
    folder.mkdir(parents=True, exist_ok=True)
    fp_path = folder / f"{split}_fp.npy"
    if fp_path.is_file() and len(np.load(fp_path)) == len(smiles):
        fp = np.load(fp_path)
    else:
        fp = fingerprints(frame["Drug"], CONFIG["rdkit_descriptors"])
        np.save(fp_path, fp)
    want_padel, want_gin, want_minimol = needs(cfg)
    padel = gin = minimol = None
    if want_padel:
        padel = padel_matrix(smiles, folder / f"{split}_padel.npy")
        if padel.shape[1] != len(CONFIG["padel_columns"]):
            raise RuntimeError(f"{name} PaDEL width {padel.shape[1]} != {len(CONFIG['padel_columns'])}")
    if want_gin:
        gin_device = "cuda" if device == "gpu" else "cpu"
        gin = gin_matrix(smiles, folder / f"{split}_gin.npy", gin_device)
    if want_minimol:
        minimol = minimol_matrix(smiles, folder / f"{split}_minimol.npy")
    x = assemble(fp, padel, cfg["padel_index"], gin, minimol, bool(cfg["minimol_concat"]))
    if x.shape[1] != int(cfg["n_features"]):
        raise RuntimeError(f"{name} feature width {x.shape[1]} != {cfg['n_features']}")
    return x, minimol


def run_endpoint(group, name: str, cache: Path, device: str) -> dict:
    cfg = CONFIG["endpoints"][name]
    benchmark = group.get(cfg["tdc_name"])
    train_val = benchmark["train_val"]
    test = benchmark["test"]
    x_pool, mini_pool = block_matrix(name, "train_val", train_val, cfg, cache, device)
    x_test, mini_test = block_matrix(name, "test", test, cfg, cache, device)
    y_test = test["Y"].to_numpy(float)
    predictions = []
    for seed in SEEDS:
        train, _valid = group.get_train_valid_split(benchmark=cfg["tdc_name"], split_type="default", seed=seed)
        rows = row_index(train_val, train)
        x_train = x_pool[rows]
        y_train = train_val["Y"].to_numpy(float)[rows]
        print(f"[fit] {name} seed={seed} train={len(rows)} features={x_train.shape[1]} device={device}", flush=True)
        cat_pred = fit_catboost(x_train, y_train, x_test, cfg, seed, device)
        weight = float(cfg["blend_weight"])
        if weight > 0.0:
            head_pred = fit_head(mini_pool[rows], y_train, mini_test, cfg, seed)
            cat_pred = mix(cat_pred, head_pred, weight)
        predictions.append({name: cat_pred})
    summary = group.evaluate_many(predictions)
    reported = cfg["reported_leaderboard"]
    print(f"[tdc] {name} {summary[name]}  reported {reported}", flush=True)
    return {"endpoint": name, "evaluate_many": summary[name], "reported_leaderboard": reported}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Ark-ADMET on the TDC ADMET group.")
    parser.add_argument("--endpoint", help="TDC directory name, for example pgp_broccatelli")
    parser.add_argument("--all", action="store_true", help="Run all 22 endpoints")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args()
    if args.all == bool(args.endpoint):
        raise SystemExit("Choose exactly one of --endpoint NAME or --all")
    names = list(CONFIG["endpoints"]) if args.all else [args.endpoint]
    unknown = [name for name in names if name not in CONFIG["endpoints"]]
    if unknown:
        raise SystemExit(f"Unknown endpoint {unknown}. Choices: {', '.join(CONFIG['endpoints'])}")
    group = admet_group(path=str(args.cache / "tdc"))
    rows = [run_endpoint(group, name, args.cache, args.device) for name in names]
    dest = args.cache / "leaderboard_results.json"
    dest.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[wrote] {dest}", flush=True)


if __name__ == "__main__":
    main()
