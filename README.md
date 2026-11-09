# Ark-ADMET

[中文版本](README_zh.md)

**Ark-ADMET retrains one CatBoost model per Therapeutics Data Commons ADMET endpoint and scores it with the official test-set evaluator.**

The configuration is frozen. Each endpoint keeps the same fingerprint block and may add PaDEL 3D columns, a frozen GIN embedding, a frozen MiniMol embedding, and a small network trained only on that MiniMol vector. This repository runs seeds 1–5 on the official scaffold split and prints the mean and standard deviation that the TDC leaderboard form asks for.

Inference checkpoints, without the training entry point, are at [huggingface.co/Eonark/Ark-ADMET](https://huggingface.co/Eonark/Ark-ADMET). The preprint is [bioRxiv 2026.10.28.762420](https://doi.org/10.64898/2026.10.28.762420).

## What Ark-ADMET provides

- The 22 submitted endpoint settings in `configs/endpoints.json`
- A single command that loads `ADMET_Group`, fits each seed, and calls `evaluate_many`
- The shared 2,563-d fingerprint block: Morgan counts (radius 2, 1024), Avalon counts (1024), ErG (315), and 200 RDKit descriptors
- Optional PaDEL 3D, frozen `gin_supervised_masking` (300-d), and frozen MiniMol v1 (512-d) blocks, computed once and cached
- The preprint's test-set summary next to the value `evaluate_many` prints

## How Ark-ADMET differs from existing models

The comparison is the one in the [preprint](https://doi.org/10.64898/2026.10.28.762420). MapLight and MiniMol each use one configuration on all 22 endpoints. Ark-ADMET keeps one CatBoost backbone and the MapLight fingerprint block, and lets each endpoint accept or reject the same optional pieces.

| | MapLight | MiniMol | Ark-ADMET |
| --- | --- | --- | --- |
| Backbone | CatBoost | Frozen graph encoder plus a shallow head | CatBoost |
| Features on every endpoint | Morgan, Avalon, ErG, and 200 RDKit descriptors. A second entry also appends one frozen graph embedding | The same 512-d encoder | The same MapLight fingerprint block |
| What can change by endpoint | The published recipe is shared | The published recipe is shared | PaDEL 3D column count, frozen GIN, frozen MiniMol inside the tree, blend weight with a width-128 MiniMol network, and the tree settings |
| How the settings are chosen | One published recipe for all 22 endpoints | One published recipe for all 22 endpoints | Take the first training run and hold out a slice grouped by molecular scaffold. Replace the current setting only when that slice scores better. Score the official test set only after the choice is fixed; the test set does not take part in the choice |

On the preprint's frozen copy of the public leaderboard, the 22 models are rank 1 on 7 of the 21 endpoints that can be ranked. Three of those seven are ahead of the previous best entry: microsome clearance (Spearman 0.634 versus 0.630), PPBR (MAE 7.349 versus 7.440), and CYP3A4 inhibition (AUPRC 0.917 versus 0.916). Four tie the previous best: solubility (0.741), HIA (0.993), Ames (0.871), and CYP2C9 inhibition (0.859). Five more are rank 2 or 3. Nine are rank 4 or worse. No single optional block is present in every rank-1 endpoint. The live board can move after that snapshot.

## 2-minute quickstart

### Prerequisites

- Python 3.11
- Network access the first time PyTDC downloads the ADMET tables
- A GPU for the submitted setting. `--device cpu` runs the same hyperparameters on 16 threads and can move the third decimal

Endpoints that keep PaDEL, GIN, or MiniMol also need Java, `padelpy`, `dgllife`, `dgl`, `torch`, or `minimol`. See `requirements-optional.txt`. `pgp_broccatelli` uses fingerprints only.

### 1) Install

```bash
git clone https://github.com/AIohaaaa/Ark-ADMET.git
cd Ark-ADMET
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Score one endpoint

```bash
python reproduce_tdc.py --endpoint pgp_broccatelli --device gpu
```

### 3) Read the leaderboard line

The command prints `evaluate_many` for that endpoint and writes `cache/leaderboard_results.json`. Each seed is rounded to three decimal places. The reported pair is the mean and population standard deviation of those five numbers, again rounded to three decimal places.

## Full group

```bash
python reproduce_tdc.py --all --device gpu
```

PaDEL, GIN, and MiniMol are computed the first time an endpoint needs them, then read from `./cache`.

For every endpoint and every seed in `{1, 2, 3, 4, 5}` the script:

1. Loads the official `ADMET_Group` train-validation and test tables.
2. Calls `get_train_valid_split(split_type="default", seed=seed)`.
3. Fits CatBoost on that training portion only. The official validation rows are not used for fitting or early stopping.
4. Adds the frozen optional blocks and, when `blend_weight > 0`, mixes in a width-128 network trained on MiniMol alone.
5. Scores the official test table with `group.evaluate_many`.

Submitted hardware: one NVIDIA L20, CatBoost `task_type="GPU"`, `devices="0"`, `gpu_ram_part=0.85`. The MiniMol head runs on CPU.

## Reported test summary

Preprint values stored as `reported_leaderboard`. Lower MAE is better. Higher ROC-AUC, AUPRC, and Spearman are better.

| Endpoint | Metric | Mean ± std |
| --- | --- | --- |
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

## Notes and limits

- The Chemprop lipophilicity model (MAE 0.412) is a separate model. It is not in this CatBoost family and is not trained here.
- Optional blocks are not bundled. The script builds them from the public PaDEL, DGL-Life, and MiniMol releases.
- Two solubility test molecules have no MiniMol vector. Those rows use the CatBoost prediction alone.
- CYP2D6 substrate is the only endpoint with `scale_pos_weight` other than 1. The value is 3.
- VDss, half-life, and both clearance endpoints are fit on a log target and inverted before scoring.
- CPU and GPU CatBoost runs are not bitwise identical.

## Citation

Yuzhu Shi, Haoxi Luo, Feng Liu, Simiao Zhao. Ark-ADMET: endpoint-specific selection of molecular representations and hyperparameters for gradient-boosted ADMET models under the Therapeutics Data Commons benchmark protocol. bioRxiv 2026.10.28.762420. https://doi.org/10.64898/2026.10.28.762420

## License

Apache-2.0. See [LICENSE](LICENSE).
