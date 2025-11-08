# DHRCL-DRP: Dual-view Hypergraph Rebalanced Contrastive Learning for Drug Response Prediction

## Introduction

Accurate drug response prediction (DRP) is crucial for advancing personalized cancer therapeutics. While existing graph-based approaches have shown considerable potential in this field, they still face persistent challenges in effectively integrating multi-omics data, adequately capturing complex high-order relationships within cell lines or drugs, and mitigating data scarcity and class imbalance.

To address these critical challenges, we propose **DHRCL-DRP**, a novel **D**ual-view **H**ypergraph **R**ebalanced **C**ontrastive **L**earning framework for **D**rug **R**esponse **P**rediction. DHRCL-DRP significantly advances the state-of-the-art by robustly integrating multi-omics data, capturing intricate high-order associations, and effectively handling sparse annotations and class imbalance, leading to superior prediction performance.

## Key Contributions

- A novel framework termed DHRCL-DRP is proposed for drug response prediction.
- A multi-granularity hierarchical attention module is devised to integrate multi-omics data.
- A dual-view hypergraph construction method is presented to capture high-order relations.
- A rebalanced contrastive learning strategy is introduced to mitigate data scarcity.
- DHRCL-DRP significantly outperforms state-of-the-art methods on two public datasets.

## Methodology

DHRCL-DRP adopts a multi-faceted approach to address the limitations of current DRP methods:

1.  **Multi-granularity Hierarchical Attention Module:** We utilize a sophisticated attention mechanism to robustly integrate heterogeneous multi-omics profiles of cell lines, discerning their interrelations and relative importance. This module captures multi-granularity features at both intra-omics and inter-omics levels.
2.  **Dual-view Hypergraph Construction:** We introduce a novel method to explicitly capture complex high-order associations from both sensitive and resistant views through hypergraphs. This allows for a deeper understanding of multi-way interactions among cell lines or drugs.
3.  **Rebalanced Contrastive Learning Strategy:** To mitigate sparse annotations and class imbalance, a carefully designed rebalanced contrastive learning strategy optimizes feature embeddings. This strategy enhances discriminability, alleviates over-smoothing, and ensures fair learning for minority classes.

The overall framework combines these robust feature representations with a two-layer GCN for cell line-drug response association information and a normalized inner product decoder to reconstruct the sensitivity indicator matrix.

## Experimental Results

Extensive experiments on two public datasets, GDSC and CCLE, demonstrate that DHRCL-DRP significantly outperforms six representative baseline methods across all evaluation metrics.

**Performance Comparison on GDSC and CCLE Datasets:**

| Dataset | Method        | AUC (%)          | AUPR (%)         | F1 (%)           | ACC (%)          | MCC (%)          |
| :------ | :------------ | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| GDSC    | DeepCDR       | 80.40 ± 0.43     | 40.53 ± 0.57     | 42.46 ± 0.80     | 82.45 ± 0.59     | 43.44 ± 0.94     |
|         | GraOmicDRP    | 74.46 ± 0.58     | 38.79 ± 1.47     | 42.33 ± 0.49     | 81.87 ± 0.60     | 34.75 ± 0.33     |
|         | GraphCDR      | 81.21 ± 0.83     | 40.38 ± 0.23     | 41.52 ± 0.15     | **88.35 ± 0.10** | **44.31 ± 0.21** |
|         | MOFGCN        | 83.31 ± 0.23     | 49.37 ± 0.54     | 46.81 ± 0.75     | 87.64 ± 0.63     | 41.70 ± 0.88     |
|         | RedCDR        | **84.73 ± 0.53** | **50.06 ± 0.13** | **47.86 ± 0.11** | 87.83 ± 0.54     | 43.22 ± 0.12     |
|         | HRLCDR        | 76.80 ± 0.14     | 37.79 ± 0.25     | 42.56 ± 0.53     | 41.42 ± 0.18     | 33.27 ± 0.02     |
|         | **DHRCL-DRP** | **85.61 ± 0.15** | **52.55 ± 0.37** | **49.22 ± 0.27** | **89.78 ± 0.13** | **49.84 ± 0.35** |
| CCLE    | DeepCDR       | 92.46 ± 0.56     | 84.39 ± 1.48     | 80.24 ± 1.80     | 90.24 ± 0.67     | 75.16 ± 2.19     |
|         | GraOmicDRP    | 90.57 ± 1.62     | 86.52 ± 2.79     | 81.74 ± 2.34     | 91.25 ± 0.89     | 77.68 ± 2.88     |
|         | GraphCDR      | 93.78 ± 0.97     | 79.85 ± 0.41     | 78.46 ± 0.35     | 91.30 ± 0.46     | 73.39 ± 2.17     |
|         | MOFGCN        | 93.41 ± 0.33     | 85.68 ± 0.38     | 79.33 ± 0.27     | 91.82 ± 0.60     | 77.45 ± 0.94     |
|         | RedCDR        | **95.76 ± 0.68** | **86.68 ± 1.60** | 80.08 ± 1.08     | 92.29 ± 0.51     | 78.03 ± 1.37     |
|         | HRLCDR        | 95.26 ± 0.83     | 86.26 ± 1.01     | **81.04 ± 1.61** | **92.72 ± 0.63** | **79.70 ± 0.37** |
|         | **DHRCL-DRP** | **96.25 ± 0.07** | **87.93 ± 0.17** | **81.81 ± 0.12** | **93.74 ± 0.56** | **81.82 ± 0.34** |

_All bold values indicate the best performance, and underlined values indicate the second-best performance._

DHRCL-DRP achieves average AUCs of 85.61% and 96.25% on the GDSC and CCLE datasets, respectively, demonstrating its superior performance. Ablation studies confirm the positive contribution of each module within the DHRCL-DRP framework.

## Datasets

The datasets used in this study are GDSC and CCLE. Cell line multi-omics data (gene expression, genomic mutation, DNA methylation) are retrieved from CCLE. Cancer cell line-drug response data (IC50 values or activity area values) are binarized as described in the paper. Drug chemical structures (SMILES strings) are sourced from PubChem and converted into molecular graphs using DeepChem.

All datasets are available at: (https://github.com/mhxu1998/RedCDR)
