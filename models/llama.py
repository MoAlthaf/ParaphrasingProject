import os
import numpy as np
import time
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path
from src.utils.sql_utils import extract_schema
from multiprocessing import set_start_method

load_dotenv()  # Load environment variables from .env file

# Set threading layer early
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['HF_TOKEN'] = os.getenv("HF_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HF_API_KEY")


# === Change model and tokenizer here ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


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

    sys_msg = (
        "You are a helpful assistant for paraphrasing natural language questions that will be converted into SQL queries. Rewrite the question using different words or phrasing, but make sure the meaning and logic are exactly the same so that both the original and paraphrased versions would generate the same SQL query.\n\n"
        "- Keep all columns, tables, filters, and conditions unchanged.\n"
        "- Do not add, remove, or change any information.\n"
        "- Do not change quantifiers (like 'each', 'every', 'distinct', 'any').\n"
        "- Do not change time or comparison logic (e.g., 'after 2013' must stay 'after 2013').\n"
        "- Do not expand or shorten abbreviations keep it unchanged.\n\n"
        "- Do NOT change the casing of string values. For example, keep 'france' as 'france' — do not convert to 'FRANCE' or 'France'."
        "Return only the paraphrased question as plain text, with no explanation or extra text."
    )

    usr_msg = (
        f"Here is the table schema for context:\n{schema}\n\n"
        f"Original question:\n{sentence}"
    )

    chat_input = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_msg},
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


def generate_sql():
    pass

def regenerate_paraphrase(question:str, schema:str) -> str:
    return question

if __name__=="__main__":
    pass


