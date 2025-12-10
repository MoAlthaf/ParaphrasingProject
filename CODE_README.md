**Project Overview**
- **Purpose:**: A pipeline to evaluate paraphrasing quality for NL→SQL tasks. The code paraphrases natural-language questions, generates SQL from both original and paraphrased questions using multiple LLM backends, compares generated SQL against ground-truth, and aggregates results.
- **Main flow:**: prepare dataset → paraphrase questions → generate SQL (per model) → compare SQL → aggregate results.

**Models (current defaults)**
- **LLaMA:**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (used in `models/llama.py`)
- **Mistral:**: `mistralai/Mistral-7B-Instruct-v0.3` (used in `models/mistral.py`)
- **Qwen:**: `XGenerationLab/XiYanSQL-QwenCoder-7B-2504` (used in `models/qwen.py`)

All model files use `vllm.LLM` + `transformers.AutoTokenizer` and expect a Hugging Face API key.

**Important environment variables & dependencies**
- **Hugging Face token:**: set `HF_API_KEY` (the code reads this into `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`). The project uses `python-dotenv` so you can put `HF_API_KEY=...` in a `.env` file or export it in the shell.
- **Dependencies:**: see `requirements.txt`. Key packages: `vllm`, `transformers`, `pandas`, `python-dotenv`.

**GPU & Hardware Requirements**

The project uses `vllm` for efficient LLM inference. Even though models run **sequentially** (not in parallel), each model occupies GPU memory during its run. Here's a rough estimation:

**Model Memory Requirements (per model, loaded individually):**
- **LLaMA 3.1-8B**: ~16–18 GB VRAM (8B params × 2 bytes per param + overhead)
- **Mistral 7B**: ~14–16 GB VRAM (7B params × 2 bytes + overhead)
- **Qwen 7B (XiYanSQL)**: ~14–16 GB VRAM (7B params × 2 bytes + overhead)

**Total GPU memory needed:**
- **Minimum (run one model at a time, sequential)**: **18 GB** (enough for the largest model, LLaMA 3.1-8B)
  - Models load and unload one at a time, so you only need enough VRAM for the largest model.
- **Recommended (headroom for batching & overhead)**: **20–24 GB** 
  - Leaves margin for batch processing, gradient computation, and vLLM's KV cache.


**vLLM Configuration Notes:**
- Current code uses `max_model_len=2048` and `tensor_parallel_size=1` (single GPU, no sharding).
- `tensor_parallel_size=1` means no model parallelism; the entire model must fit on one GPU.
- If you have multi-GPU setup, you can increase `tensor_parallel_size` to shard a single model across GPUs, but models still run sequentially.
- Batch size in vLLM is controlled implicitly; for smaller datasets, batching is limited by the number of rows processed at once.



**Data source**
- The project expects its dataset under `data/` (see `data/database/` for per-database folders). The dataset used for experiments is the [Spyder dataset](https://yale-lily.github.io/spider).

**How the code is organized**
- **`main.py`**: Orchestrator. Controls: dataset creation, paraphrasing (LLaMA only), NL→SQL runs per model (all 3 models), and evaluation/aggregation.
- **`models/*.py`**: Model-specific wrappers for LLM access and batch NL→SQL evaluation (`generate_sql`, `generate_sql_from_dataframe`). `models/prompt_templates.py` contains the system prompts.
  - **Paraphrasing:** Only `models/llama.py` provides `paraphrase_sentence()`. This runs once to create the paraphrased dataset.
  - **NL→SQL:** All three models (`llama.py`, `qwen.py`, `mistral.py`) generate SQL from both original and paraphrased questions.
- **`src/`**: utilities (dataset preparation, SQL extraction/comparison, paraphrase scoring, logging).
- **Outputs:**: `result/` (per-model CSVs + `results.csv` + `structured_result.json`) and `logs/`.

**How to run**
Prerequisites:
- Install dependencies: `pip install -r requirements.txt`.
- Provide a Hugging Face API key via one of these options:
  - Create a `.env` file at project root with:

    ```powershell
    HF_API_KEY=your_hf_api_key_here
    ```

  - Or set it in PowerShell for the session:

    ```powershell
    $env:HF_API_KEY = 'your_hf_api_key_here'
    ```

Running `main.py` directly
- The easiest default run is to execute `main.py` which runs the `main()` call defined at the bottom of the file. By default the script runs with these settings (as written in `if __name__ == "__main__"`): it will only run the evaluation step (`evaluate=True`) and skip model runs. To run the full pipeline you can either edit the flags in `main.py` or call `main()` programmatically from the command line.
- **Pipeline stages:**
  1. **Dataset prep** (optional, `dataset_force`) — generates initial dataset.
  2. **Paraphrasing** (optional, `paraphrasing_force`) — **LLaMA only** generates paraphrases & scores them.
  3. **NL→SQL** (optional, `nl2sql_force`) — **all selected models** (LLaMA, Qwen, Mistral) generate SQL queries.
  4. **Evaluation** (optional, `evaluate`) — merge results from all 3 models and create aggregated outputs.

Run examples (PowerShell)
- Run the default `__main__` invocation (no model runs by default):

```powershell
python main.py
```

- Run the pipeline programmatically (choose which steps/models to run). This avoids editing the file — run a one-liner that imports and calls `main` with chosen arguments:

```powershell
python -c "from main import main; main(dataset_force=$True, paraphrasing_force=$True, nl2sql_force=$True, evaluate=$True, run_llama=$False, run_qwen=$False, run_mistral=$True, threshold=0.7, max_retries=1)"
```

Note: PowerShell uses `$True`/`$False` in the inline call above; if that causes issues, you can use `python -c` with plain `True`/`False` depending on quoting, or create a small runner script.

What each `main()` parameter controls:
- **`dataset_force`**: bool — regenerate interim dataset (`data/interim/generated_queries.csv`).
- **`paraphrasing_force`**: bool — re-run paraphrasing (LLaMA only) and overwrite `data/processed/output_paraphrased.csv`.
- **`nl2sql_force`**: bool — run NL→SQL generation step for all selected models.
- **`evaluate`**: bool — merge per-model outputs and create `result/results.csv` and `result/structured_result.json`.
- **`run_llama`, `run_qwen`, `run_mistral`**: bool — which model(s) to run during NL→SQL stage. All three can be used independently.
- **`threshold`**: float — paraphrase acceptance threshold used by `score_paraphrase` (only applies to LLaMA paraphrasing stage).
- **`max_retries`**: int — number of retries if paraphrase score is below threshold (only applies to LLaMA paraphrasing stage).

Output locations
- **Per-model CSVs:**: `result/llama_results.csv`, `result/qwen_results.csv`, `result/mistral_results.csv` (created by `models/*` runners).
- **Aggregated results:**: `result/results.csv` and `result/structured_result.json`.
- **Logs:**: `logs/main.log`, `logs/llama.log`, `logs/qwen.log`, `logs/mistral.log` (created via `src.utils.logger.setup_logger`).

**Notes**
- **Paraphrasing is LLaMA-only**: Only `models/llama.py` implements `paraphrase_sentence()`. The paraphrasing stage (step 2 in the pipeline) always uses LLaMA, regardless of which models you select for NL→SQL.
- `models/qwen.py` and `models/mistral.py` do not include `paraphrase_sentence()` — they are used only for NL→SQL generation.
- `regenerate_paraphrase()` in `models/llama.py` is a stub that currently returns the original question; you may want to implement a retry prompt or a fallback model for low-scoring paraphrases.
- Models use `trust_remote_code=True` when loading tokenizers — be careful and pin or audit models if running in sensitive environments.
- Make sure the `data/database/` folder contains the expected DB folders (one per dataset DB) with `*.sqlite` files so `src.utils.sql_utils.extract_schema` can find schemas.
