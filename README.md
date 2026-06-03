# MESC

Memory-Enhanced Specialist Consultation (MESC) for difficulty-aware medical diagnosis with large language models.

## Introduction

Medical consultation with large language models often applies the same inquiry workflow to all patient cases, which can lead to redundant symptom questions and unnecessary reasoning cost. MESC addresses this issue with a memory-enhanced specialist consultation framework that adapts the consultation process according to case familiarity and diagnostic uncertainty.

MESC follows two complementary paths. For typical cases, it first retrieves similar structured case memories, verifies key symptoms, and reuses an efficient inquiry path for fast diagnosis. For atypical or uncertain cases, it routes the patient to multi-specialist collaborative inquiry, where specialist agents ask targeted symptom questions and update disease confidence with collected evidence.

<p align="center">
  <img src="docs/mesc_framework.png" width="900">
</p>

<p align="center">
  <b>Figure:</b> Overview of the MESC framework. Memory-first routing quickly handles typical cases and supports multi-specialist inquiry for uncertain cases.
</p>

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The default dependency list uses `faiss-cpu` for portability. If GPU-based FAISS is required, please install the corresponding `faiss-gpu` package in your local CUDA environment.

The code was developed for PyTorch-based LLM inference and MESC consultation evaluation. Please prepare the corresponding LLM checkpoints locally before running experiments.

## Project Structure

```text
MESC/
├── data/                  # DXY, GMD, CMD datasets and turn-level data
├── outputs/               # Policies, Golden Records, adapters, and generated results
├── run_consultation.py    # Main inference entry
├── calibrate_diagnostic_llm.py
├── run_mesc_inference.sh
└── calibrate_mesc_llm.sh
```

## Model Path

The code does not hard-code local absolute model paths. By default, models are loaded from:

```text
./models/<model-directory>
```

You can also specify a local model root or a specific model path:

```bash
export MESC_MODEL_ROOT=/path/to/models
# or
export MESC_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
```

For example, `qwen2.5-7b-instruct` maps to `Qwen2.5-7B-Instruct` under the model root.

## Data

Place the benchmark datasets under `data/`:

```text
data/DXY/
data/GMD/
data/CMD/
```

Each dataset directory should contain `train.json`, `dev.json`, `test.json`, `empirical_knowledge.json`, `disease_corpurs.txt`, and `symptom_corpurs.txt`.

## How to Run

Run MESC inference from the project root:

```bash
bash run_mesc_inference.sh
```

You can edit `run_mesc_inference.sh` to change the dataset, LLM backbone, adapter checkpoint, policy path, and consultation parameters.
