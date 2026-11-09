# Ark-ADMET

Per-endpoint CatBoost models for the [Therapeutics Data Commons](https://tdcommons.ai/benchmark/admet_group/overview/) ADMET benchmark group.

This repository retrains the configuration reported in the preprint and scores it with TDC `evaluate_many`. That is the number the leaderboard form asks for. Released checkpoints, for inference only, are at [huggingface.co/Eonark/Ark-ADMET](https://huggingface.co/Eonark/Ark-ADMET).

## Citation

Yuzhu Shi, Haoxi Luo, Feng Liu, Simiao Zhao. Ark-ADMET: endpoint-specific selection of molecular representations and hyperparameters for gradient-boosted ADMET models under the Therapeutics Data Commons benchmark protocol. bioRxiv 2026.10.28.762420. https://doi.org/10.64898/2026.10.28.762420

## What is reproduced

For each of the 22 endpoints and for seeds 1–5:

1. Load the official TDC `ADMET_Group` tables.
2. Call `get_train_valid_split(split_type="default", seed=seed)`.
3. Fit CatBoost on that seed's training rows only. The official validation rows are not fit and are not used for early stopping.
4. Where the frozen configuration says so, add a subset of 431 PaDEL 3D columns, a frozen 300-d `gin_supervised_masking` embedding, a frozen 512-d MiniMol v1 embedding inside the tree, and a convex combination with a one-hidden-layer network (width 128) trained on MiniMol alone.
5. Score the official test set and aggregate with `group.evaluate_many`.

`configs/endpoints.json` is the frozen configuration: tree depth, number of trees, learning rate, L2, PaDEL column indices, GIN, MiniMol, blend weight, and the CYP2D6-substrate positive-class weight. Feature and hyperparameter selection was finished before these values were written down. This repository does not search them again.

Every endpoint uses the same 2,563 fingerprint columns: Morgan counts (radius 2, 1,024), Avalon counts (1,024), ErG (315), and 200 RDKit descriptors.

The leaderboard numbers were trained on one NVIDIA L20, CatBoost `task_type="GPU"`, `devices="0"`, `gpu_ram_part=0.85`. A CPU run uses the same hyperparameters with `thread_count=16` and can differ in the third decimal.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Endpoints that keep extra blocks also need `requirements-optional.txt`:

| Block | Extra packages | Also required |
|---|---|---|
| PaDEL 3D | `padelpy` | Java on `PATH` |
| GIN | `dgllife`, `dgl`, `torch` | the public `gin_supervised_masking` weights via DGL-Life |
| MiniMol | `minimol`, `torch` | MiniMol v1 weights, downloaded by that package |

Fingerprint-only endpoints need none of those. `pgp_broccatelli` is one of them.

## Reproduce

From this directory:

```bash
# One fingerprint-only endpoint. Prints the evaluate_many mean and standard deviation.
python reproduce_tdc.py --endpoint pgp_broccatelli --device gpu

# Every endpoint. PaDEL, GIN, and MiniMol are computed the first time an endpoint needs them
# and then read from ./cache.
python reproduce_tdc.py --all --device gpu
```

Use `--device cpu` on a machine without a GPU. Results are written to `cache/leaderboard_results.json`. Each record has `evaluate_many`, which is the value to paste into the TDC form, and `reported_leaderboard`, the value from the preprint.

`evaluate_many` rounds each seed to three decimal places, then reports the mean and population standard deviation of those five numbers, rounded to three decimal places.

## Reported test summary

These are the preprint values (`reported_leaderboard` in the config). Lower MAE is better. Higher ROC-AUC, AUPRC, and Spearman are better.

| Endpoint | Metric | Mean ± std |
|---|---|---|
| Caco2_Wang | MAE | 0.301 ± 0.009 |
| HIA_Hou | AUROC | 0.993 ± 0.002 |
| Pgp_Broccatelli | AUROC | 0.930 ± 0.008 |
| Bioavailability_Ma | AUROC | 0.744 ± 0.026 |
| Lipophilicity_AstraZeneca | MAE | 0.479 ± 0.007 |
| Solubility_AqSolDB | MAE | 0.741 ± 0.008 |
| BBB_Martins | AUROC | 0.917 ± 0.006 |
| PPBR_AZ | MAE | 7.349 ± 0.025 |
| VDss_Lombardo | Spearman | 0.681 ± 0.017 |
| CYP2C9_Veith | AUPRC | 0.859 ± 0.003 |
| CYP2D6_Veith | AUPRC | 0.788 ± 0.004 |
| CYP3A4_Veith | AUPRC | 0.917 ± 0.002 |
| CYP2C9_Substrate_CarbonMangels | AUPRC | 0.385 ± 0.018 |
| CYP2D6_Substrate_CarbonMangels | AUPRC | 0.692 ± 0.013 |
| CYP3A4_Substrate_CarbonMangels | AUROC | 0.641 ± 0.023 |
| Half_Life_Obach | Spearman | 0.548 ± 0.042 |
| Clearance_Hepatocyte_AZ | Spearman | 0.448 ± 0.015 |
| Clearance_Microsome_AZ | Spearman | 0.634 ± 0.006 |
| LD50_Zhu | MAE | 0.570 ± 0.013 |
| hERG | AUROC | 0.864 ± 0.008 |
| AMES | AUROC | 0.871 ± 0.001 |
| DILI | AUROC | 0.891 ± 0.016 |

A separate Chemprop model for lipophilicity (MAE 0.412) is not part of this CatBoost family and is not trained here.

## License

Apache-2.0. See [LICENSE](LICENSE).
