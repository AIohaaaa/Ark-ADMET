# Ark-ADMET

[English](README.md)

**Ark-ADMET 为 Therapeutics Data Commons 的每个 ADMET 端点各重训一棵 CatBoost，并用官方测试集评价器打分。**

配置已经冻结。每个端点共用同一段指纹，并按端点决定是否加上 PaDEL 三维列、冻结的 GIN 向量、冻结的 MiniMol 向量，以及一个只看 MiniMol 的小网络。本仓库在官方骨架划分上跑种子 1–5，打印 TDC 榜单表单要的均值和标准差。

只做推理的检查点在 [huggingface.co/Eonark/Ark-ADMET](https://huggingface.co/Eonark/Ark-ADMET)。预印本是 [bioRxiv 2026.10.28.762420](https://doi.org/10.64898/2026.10.28.762420)。

## 这个仓库提供什么

- `configs/endpoints.json` 里 22 个端点的提交配置
- 一条命令：读取 `ADMET_Group`，按种子拟合，并调用 `evaluate_many`
- 共用的 2563 维指纹：Morgan 计数（半径 2，1024）、Avalon 计数（1024）、ErG（315）、200 个 RDKit 描述符
- 按需计算并缓存的 PaDEL 三维、冻结的 `gin_supervised_masking`（300 维）和冻结的 MiniMol v1（512 维）
- 预印本里的测试集汇总，以及 `evaluate_many` 实际打印的数

## 适合谁 / 不适合谁

**适合：**

- 要按 TDC 官方骨架划分，把 22 个端点各重训五个种子，并得到榜单用的均值和标准差
- 要核对每个端点最终留下的树、PaDEL 列、GIN、MiniMol 和融合系数
- 先跑一个只用指纹的端点，确认流程通了再跑其余端点

**不适合：**

- 要在自己的活性数据、或另一套划分上训练
- 要一个 22 个端点共用同一套特征和同一组超参数的模型
- 要一个现成的预测接口。给新 SMILES 打分用 [Hugging Face 检查点](https://huggingface.co/Eonark/Ark-ADMET)

## 本仓库和检查点发布的差别

| | 本仓库 | Hugging Face 检查点 |
| --- | --- | --- |
| 用途 | 重训并给 TDC 测试集打分 | 用已发布的权重预测 |
| 榜单数字 | `evaluate_many` 的均值和标准差 | 这里不计算 |
| 额外特征 | 端点需要时现算 | 调用者自己提供 PaDEL、GIN 和 MiniMol |
| 超参数搜索 | 不包含 | 不包含 |

## 两分钟上手

### 环境

- Python 3.11
- 第一次运行时 PyTDC 需要联网下载 ADMET 表
- 提交时用的是 GPU。`--device cpu` 用同样的超参数和 16 个线程，第三位小数可能不同

保留了 PaDEL、GIN 或 MiniMol 的端点还需要 Java、`padelpy`、`dgllife`、`dgl`、`torch` 或 `minimol`。见 `requirements-optional.txt`。`pgp_broccatelli` 只用指纹。

### 1）安装

```bash
git clone https://github.com/AIohaaaa/Ark-ADMET.git
cd Ark-ADMET
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2）给一个端点打分

```bash
python reproduce_tdc.py --endpoint pgp_broccatelli --device gpu
```

### 3）看榜单那一行

命令会打印该端点的 `evaluate_many`，并写入 `cache/leaderboard_results.json`。每个种子先四舍五入到三位小数，再取这五个数的平均和总体标准差，结果再保留三位小数。

## 跑完全部端点

```bash
python reproduce_tdc.py --all --device gpu
```

PaDEL、GIN 和 MiniMol 在某个端点第一次用到时计算，之后从 `./cache` 读取。

每个端点、每个种子 `{1, 2, 3, 4, 5}`：

1. 读取官方 `ADMET_Group` 的训练–验证表和测试表。
2. 调用 `get_train_valid_split(split_type="default", seed=seed)`。
3. CatBoost 只拟合该次划分的训练部分。官方验证集不参与拟合，也不用来早停。
4. 按冻结配置加上可选特征；`blend_weight > 0` 时，再与只看 MiniMol、宽度 128 的网络混合。
5. 用 `group.evaluate_many` 给官方测试表打分。

提交时的硬件：一块 NVIDIA L20，CatBoost `task_type="GPU"`、`devices="0"`、`gpu_ram_part=0.85`。MiniMol 小网络在 CPU 上训练。

## 预印本中的测试结果

这些数存在配置的 `reported_leaderboard` 里。MAE 越低越好。ROC-AUC、AUPRC 和 Spearman 越高越好。

| 端点 | 指标 | 均值 ± 标准差 |
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

## 限制

- 亲脂性的 Chemprop（MAE 0.412）是另一条模型，不在这组 CatBoost 里，这里也不训练。
- 可选特征块不随仓库发布。脚本用公开的 PaDEL、DGL-Life 和 MiniMol 现算。
- 溶解度测试集有两个分子没有 MiniMol 向量，这两行退回 CatBoost。
- 只有 CYP2D6 底物的 `scale_pos_weight` 不是 1，取值是 3。
- 分布容积、半衰期和两个清除率端点在对数目标上拟合，打分前变回原尺度。
- CPU 和 GPU 上的 CatBoost 结果不是逐位相同的。

## 引用

Yuzhu Shi, Haoxi Luo, Feng Liu, Simiao Zhao. Ark-ADMET: endpoint-specific selection of molecular representations and hyperparameters for gradient-boosted ADMET models under the Therapeutics Data Commons benchmark protocol. bioRxiv 2026.10.28.762420. https://doi.org/10.64898/2026.10.28.762420

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)。
