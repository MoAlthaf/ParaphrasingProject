# ParaphrasingProject

A pipeline to evaluate paraphrasing quality for NL→SQL tasks. The code paraphrases natural-language questions, generates SQL from both original and paraphrased questions using multiple LLM backends (LLaMA, Mistral, Qwen), compares generated SQL against ground-truth, and aggregates results.

## Quick Start

**Setup:**
```bash
pip install -r requirements.txt
```

**Environment:**
Set your Hugging Face API key:
```powershell
$env:HF_API_KEY = 'your_hf_api_key_here'
# or create .env file: HF_API_KEY=your_key
```

**Run:**
```powershell
python main.py
```

For detailed setup, GPU requirements, and how to customize pipeline parameters, see [`CODE_README.md`](CODE_README.md).

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[`CODE_README.md`](CODE_README.md)** | Project overview, setup, GPU requirements, how to run with parameters |
| **[`FUNCTIONS_README.md`](FUNCTIONS_README.md)** | Detailed function signatures and behavior for `main.py` and model modules |
| **[`UTILS_README.md`](UTILS_README.md)** | Database utilities, schema extraction, SQL comparison |

---

## Project Structure

```
ParaphrasingProject/
├── main.py                          # Orchestrator: controls pipeline stages
├── models/
│   ├── llama.py                     # LLaMA model wrapper (paraphrasing + NL→SQL)
│   ├── mistral.py                   # Mistral model wrapper (NL→SQL only)
│   ├── qwen.py                      # Qwen model wrapper (NL→SQL only)
│   └── prompt_templates.py          # System prompts for paraphrasing & SQL generation
├── src/
│   ├── prepare_dataset.py           # Dataset preparation
│   └── utils/
│       ├── sql_utils.py             # Database operations, schema extraction, SQL comparison
│       ├── logger.py                # Logging setup
│       ├── paraphrase_score.py      # Paraphrase quality scoring
│       └── __init__.py
├── data/
│   ├── database/                    # SQLite databases (one per dataset DB)
│   ├── interim/                     # Generated queries before paraphrasing
│   └── processed/                   # Paraphrased questions + scores
├── result/                          # Model outputs, aggregated results, structured JSON
├── logs/                            # Log files per stage
├── CODE_README.md                   # Setup & configuration guide
├── FUNCTIONS_README.md              # Function reference
├── UTILS_README.md                  # Utility function reference
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Pipeline Overview

The project runs in 4 stages (all optional):

1. **Dataset Preparation** — Generate initial dataset from source.
2. **Paraphrasing** — LLaMA generates paraphrases and scores them (`paraphrasing_force` flag).
3. **NL→SQL Generation** — All selected models (LLaMA, Mistral, Qwen) generate SQL from original and paraphrased questions (`nl2sql_force` flag).
4. **Evaluation** — Merge results from all models and create aggregated outputs (`evaluate` flag).

**Example runs:**
- Paraphrase only: `main(paraphrasing_force=True, nl2sql_force=False)`
- Run Mistral NL→SQL only: `main(nl2sql_force=True, run_mistral=True, run_llama=False, run_qwen=False)`
- Full pipeline: `main(paraphrasing_force=True, nl2sql_force=True, evaluate=True)`

See [`CODE_README.md`](CODE_README.md) for detailed parameter documentation.

---

## Models

- **LLaMA 3.1-8B-Instruct** — Used for paraphrasing + NL→SQL
- **Mistral 7B-Instruct** — Used for NL→SQL only
- **Qwen (XiYanSQL)** — Used for NL→SQL only

All models use `vllm` for efficient inference. Requires **18–24 GB VRAM**. See [`CODE_README.md`](CODE_README.md) for detailed GPU requirements.

---

## Dataset

The project uses the [Spider dataset](https://yale-lily.github.io/spider). Expected structure:
```
data/database/
├── academic/
│   └── academic.sqlite
├── airline/
│   └── airline.sqlite
└── ... (100+ databases)
```

---

## Key Files at a Glance

| File | Role |
|------|------|
| `main.py` | Pipeline orchestrator; controls stages and model selection |
| `models/llama.py` | LLaMA wrapper (only module with paraphrasing) |
| `models/mistral.py`, `qwen.py` | NL→SQL generation (no paraphrasing) |
| `src/utils/sql_utils.py` | Database I/O, schema extraction, SQL execution & comparison |
| `result/results.csv` | Merged results from all models |
| `result/structured_result.json` | Structured output with correctness flags |

---

## Need Help?

- **How do I run the pipeline?** → [`CODE_README.md`](CODE_README.md)
- **What does `main()` do?** → [`FUNCTIONS_README.md`](FUNCTIONS_README.md)
- **How does SQL comparison work?** → [`UTILS_README.md`](UTILS_README.md)
- **What models are available?** → This file or [`CODE_README.md`](CODE_README.md)

---

Last updated: December 2025
