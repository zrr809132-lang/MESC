# MESC

Memory-Enhanced Specialist Consultation (MESC) for difficulty-aware medical diagnosis with large language models.

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
