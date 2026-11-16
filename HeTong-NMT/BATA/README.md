# BATA: Behavior-Aware Terminology Adaptation

**BATA** is a framework for improving specialized terminology translation in low-resource Tibetan–Chinese machine translation. It uses a translation model's observed behavior on terminology examples to guide lightweight model adaptation, and retrieves bilingual terms at inference time.

## How BATA works

BATA is built on **TC-Backbone**, a Tibetan–Chinese translation model adapted from Hy-MT2-7B using approximately 570K parallel sentence pairs. Rather than training on all terminology examples uniformly, BATA groups examples according to whether the backbone can produce the target term:

- **Stable Success:** the term is translated correctly without an explicit terminology prompt.
- **Recoverable Failure:** the term is produced correctly when the bilingual terminology is supplied.
- **Residual Failure:** the term remains incorrect after terminology conditioning.

These behavior patterns guide the training of two separate LoRA adapters—one for BO→ZH and one for ZH→BO—while the TC-Backbone remains frozen. At inference time, direction-aware retrieval finds relevant source-side terms, filters ambiguous mappings, and supplies valid bilingual terms to the corresponding adapter. If no eligible term is retrieved, the system uses the backbone without a terminology adapter.

<p align="center">
  <img src="assets/figure3_bata_framework.svg" alt="BATA framework" width="88%" />
</p>
<p align="center"><sub>BATA combines a Tibetan–Chinese translation backbone, direction-aware terminology retrieval, and behavior-aware direction-specific LoRA adaptation.</sub></p>

## Terminology resource

The terminology inventory contains **53,843 raw records**, consolidated into **45,432 unique normalized bilingual pairs**. Domain labels are available for **44,974 pairs (98.99%)**, organized into nine first-level domains and 21 second-level categories.

<p align="center">
  <img src="assets/figure1_data_construction.svg" alt="Terminology resource and adaptation-data construction" width="84%" />
</p>
<p align="center"><sub>Construction of the terminology resource, terminology-conditioned sentence data, and behavior-aware adaptation data.</sub></p>

<p align="center">
  <img src="assets/figure2_domain_distribution.svg" alt="Distribution of terminology pairs across nine first-level domains" width="48%" />
</p>
<p align="center"><sub>Distribution of the 44,974 terminology pairs with resolved domain labels.</sub></p>

The paper also reports 30,366 terminology-conditioned sentence instances and 2,677 behavior-aware instances (2,137 training and 540 development examples).

## Results at a glance

On the oracle-terminology evaluation, BATA achieves **91.25% Strict TermAcc** for BO→ZH and **83.30%** for ZH→BO. Compared with Random LoRA under matched training-example counts and optimizer updates, it improves Strict TermAcc by **4.50** and **48.95 percentage points**, and corpus BLEU by **1.76** and **3.87 points**, respectively.

Retrieval-based evaluation demonstrates the benefit of automatically supplying terminology. The paper also reports a trade-off between terminology intervention and general translation quality; see the paper for the full evaluation and analysis.

## Repository structure

```text
BATA/
├── README.md
├── assets/                 # Paper figures
├── configs/                # Training and retrieval configurations
├── requirements.txt
├── tests/
└── terminology_v2/
    ├── stage1_behavior_dataset_v5/   # Behavior-data builder
    ├── stage2_lora_training_v1/      # Direction-specific LoRA training
    ├── rag_integration_v1/            # Terminology indexing and retrieval
    ├── rag_lora_final2000/            # Prompting, inference, and evaluation
    └── unified_quality_evaluation_v1/ # Translation and terminology metrics
```

## Running the code

The recorded LoRA training environment uses Python 3.10.20, PyTorch 2.6.0+cu124, Transformers 5.14.1, PEFT 0.18.0, and Accelerate 1.14.0.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'
python -m pip install -r requirements.txt
```

Set the local model and data paths before running the scripts:

```bash
export BATA_TERMINOLOGY_ROOT="$PWD/terminology_v2"
export BATA_PROJECT_ROOT="$PWD"
export BATA_BASE_MODEL="/path/to/TC-Backbone"
```

Build the behavior-aware dataset and train the two adapters:

```bash
python terminology_v2/stage1_behavior_dataset_v5/build_behavior_dataset_v5.py
python terminology_v2/stage2_lora_training_v1/train_direction.py --direction bo2zh
python terminology_v2/stage2_lora_training_v1/train_direction.py --direction zh2bo
```

For inference, prepare the Final2000 records, retrieval outputs, TC-Backbone, and trained adapters. Run the alignment check before generation:

```bash
python terminology_v2/rag_lora_final2000/src/verify_alignment.py
python terminology_v2/rag_lora_final2000/src/run_inference.py --direction bo2zh --batch-size 32
python terminology_v2/rag_lora_final2000/src/run_inference.py --direction zh2bo --batch-size 32
```

Inference additionally requires a vLLM installation compatible with the local PyTorch/CUDA environment. The exact vLLM version used in the paper-mainline run was not retained.

## Data and checkpoints

This repository provides the implementation and configuration files. It does **not** distribute model checkpoints, LoRA adapter weights, source corpora, terminology databases, behavior/evaluation datasets, or generated retrieval results.

Full execution therefore requires authorized local copies of the TC-Backbone checkpoint, bilingual terminology files, upstream candidate pools, and evaluation/retrieval data. The upstream candidate-pool generation stages and the original TC-Backbone training manifest are not included, so the complete data-construction pipeline cannot be reproduced from raw corpora using this repository alone.

## Tests

The lightweight tests do not load model weights or require a GPU:

```bash
PYTHONPATH=terminology_v2/rag_integration_v1/src:terminology_v2/rag_integration_v1/ranking_v2/src:terminology_v2/rag_lora_final2000/src:terminology_v2/unified_quality_evaluation_v1 python -m unittest discover -s tests -v
```

## Citation

Zeshuo Liu, Jiu Sha, Yunhao Tian, Yu Weng, Zheng Liu, and Kailiang Wen. *BATA: Behavior-Aware Terminology Adaptation for Low-Resource Tibetan–Chinese Machine Translation.* Liu and Sha contributed equally; Yu Weng is the corresponding author.
