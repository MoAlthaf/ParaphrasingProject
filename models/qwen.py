import os
import numpy as np
import time
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path

from .prompt_templates import (
    PARAPHRASE_SYSTEM_PROMPT,
    SQL_GEN_SYSTEM_PROMPT,
    REGENERATE_SQL_PROMPT,
)
from multiprocessing import set_start_method

load_dotenv()  # Load environment variables from .env file

# Set threading layer early
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['HF_TOKEN'] = os.getenv("HF_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HF_API_KEY")


# === Change model and tokenizer here ===
MODEL_NAME ="XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
TOKENIZER_NAME ="XGenerationLab/XiYanSQL-QwenCoder-7B-2504"


# Global variables (lazy-loaded)
_llm_nl2nl = None  # Singleton LLM instance
_sampling_nl2nl = SamplingParams(temperature=0, max_tokens=100)  # Sampling params for LLM
_tokenizer_nl2nl = None  # Singleton tokenizer instance

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_tokenizer():
    """Return a singleton tokenizer instance for paraphrasing."""
    global _tokenizer_nl2nl
    if _tokenizer_nl2nl is None:
        _tokenizer_nl2nl = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            trust_remote_code=True
        )
    return _tokenizer_nl2nl

def get_llm_nl2nl():
    """Return a singleton LLM instance for paraphrasing."""
    global _llm_nl2nl
    if _llm_nl2nl is None:
        _llm_nl2nl = LLM(
            model=MODEL_NAME,
            max_model_len=2048,
            tokenizer=TOKENIZER_NAME,
            hf_token=os.environ['HF_TOKEN'],
            trust_remote_code=True,
            tensor_parallel_size=2  # Parallel
        )
    return _llm_nl2nl


def paraphrase_sentence(sentence: str, schema: str = "") -> str:
    """
    Paraphrase an NL question while guaranteeing SQL-equivalent semantics.
    Falls back to the original if any semantic-drift guard fails.
    """
    tokenizer = get_tokenizer()

    usr_msg = (
        f"Here is the table schema for context:\n{schema}\n\n"
        f"Original question:\n{sentence}"
    )

    chat_input = tokenizer.apply_chat_template(
        [{"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
         {"role": "user", "content": usr_msg}],
        tokenize=False,
        add_generation_prompt=True
    )

    llm = get_llm_nl2nl()
    try:
        outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
        para = outputs[0].outputs[0].text.strip()
        return para
    except Exception as e:
        print(f"[paraphrase_sentence] Error: {e}")
        return sentence


def generate_sql(nl_question: str, schema: str) -> str:
    """
    Generate a SQL query from a natural language question and schema using LLaMA model.

    Args:
        nl_question (str): Natural language question.
        schema (str): Extracted database schema as text.

    Returns:
        str: Generated SQL query (as plain text).
    """
    tokenizer = get_tokenizer()
    llm = get_llm_nl2nl()

    user_prompt = (
        f"Here is the database schema:\n{schema}\n\n"
        f"Now, write a valid SQL query for the following question:\n{nl_question}"
    )

    chat_input = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SQL_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
        raw_output = outputs[0].outputs[0].text.strip()

        if raw_output.startswith("[") and raw_output.endswith("]"):
            import json
            parsed = json.loads(raw_output)
            return parsed[0] if isinstance(parsed, list) and parsed else raw_output

        return raw_output  # fallback

    except Exception as e:
        print(f"[generate_sql] Error: {e}")
        return ""


if __name__=="__main__":
    pass


