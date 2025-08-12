from pathlib import Path
import pandas as pd
from multiprocessing import set_start_method

from src.prepare_dataset import main as prepare_dataset
from models.llama import paraphrase_sentence, regenerate_paraphrase, generate_sql as nl2sql_llama
from models.qwen import generate_sql as nl2sql_qwen
from models.gemma import generate_sql as nl2sql_gemma
from src.utils.sql_utils import extract_schema, compare_sql
from src.utils.logger import setup_logger
from src.utils.paraphrase_score import score_paraphrase

def main():
    # Configs
    THRESHOLD = 0.7
    MAX_RETRIES = 1
    PROJECT_ROOT = Path().resolve()

    # Paths
    DATA_PATH = PROJECT_ROOT / "data"
    dataset_path = DATA_PATH / "interim" / "generated_queries.csv"
    paraphrased_csv_path = DATA_PATH / "processed" / "output_paraphrased.csv"
    database_path = DATA_PATH / "database"
    result_path = PROJECT_ROOT / "result" / "results.csv"

    # Flags
    dataset_force = False
    paraphrasing_force = True
    nl2sql_force = True

    # Step 1: Prepare dataset
    if dataset_force or not dataset_path.exists():
        logger.info("Generating interim dataset...")
        prepare_dataset()
        logger.info("Interim dataset created.")
    else:
        logger.info("Interim dataset already exists. Skipping...")

    # Step 2: Paraphrasing
    if paraphrasing_force or not paraphrased_csv_path.exists():
        logger.info("Starting paraphrasing...")
        dataset_df = pd.read_csv(dataset_path)

        for i, row in dataset_df.iterrows():
            db_name = row['db_name']
            question = row['natural_language']
            db_full_path = database_path / db_name / f"{db_name}.sqlite"

            try:
                schema = extract_schema(db_path=db_full_path)
                paraphrased = paraphrase_sentence(question, schema)
                score = score_paraphrase(paraphrased=paraphrased, original=question)

                if score < THRESHOLD:
                    logger.warning(f"[Row {i}] Low paraphrasing score ({score:.2f}). Retrying...")
                    for attempt in range(MAX_RETRIES):
                        paraphrased = regenerate_paraphrase(question, schema)
                        score = score_paraphrase(question, paraphrased)
                        logger.info(f"[Row {i}] Retry {attempt + 1} score: {score:.2f}")
                        if score >= THRESHOLD:
                            break

            except Exception as e:
                logger.warning(f"[Row {i}] Paraphrasing error: {e}")
                paraphrased = question
                score = 0.0

            dataset_df.loc[i, "paraphrased_nl"] = paraphrased
            dataset_df.loc[i, "paraphrased_score"] = score

            logger.info(f"[Row {i}] Original: {question}")
            logger.info(f"[Row {i}] Paraphrased: {paraphrased}")
            logger.info(f"[Row {i}] Score: {score:.2f}")

        dataset_df.to_csv(paraphrased_csv_path, index=False)
        logger.info(f"Paraphrasing complete. Output saved to: {paraphrased_csv_path}")
    else:
        logger.info("Paraphrased dataset already exists. Skipping...")

    # Step 3: NL2SQL Evaluation
    if nl2sql_force:
        logger.info("Starting NL2SQL generation and evaluation...")
        paraphrased_df = pd.read_csv(paraphrased_csv_path)

        for i, row in paraphrased_df.iterrows():
            db_name = row['db_name']
            paraphrased_question = row['paraphrased_nl']
            original_sql = row['sql_query']
            original_question = row['natural_language']
            db_full_path = database_path / db_name / f"{db_name}.sqlite"

            try:
                # Generate SQL from paraphrased and original questions
                llama_query_para = nl2sql_llama(paraphrased_question, db_full_path)
                qwen_query_para = nl2sql_qwen(paraphrased_question, db_full_path)
                gemma_query_para = nl2sql_gemma(paraphrased_question, db_full_path)

                llama_query_original = nl2sql_llama(original_question, db_full_path)
                qwen_query_original = nl2sql_qwen(original_question, db_full_path)
                gemma_query_original = nl2sql_gemma(original_question, db_full_path)

                # Evaluate correctness
                llama_original_correct = compare_sql(db_full_path, original_sql, llama_query_original)
                qwen_original_correct = compare_sql(db_full_path, original_sql, qwen_query_original)
                gemma_original_correct = compare_sql(db_full_path, original_sql, gemma_query_original)

                llama_para_correct = compare_sql(db_full_path, original_sql, llama_query_para)
                qwen_para_correct = compare_sql(db_full_path, original_sql, qwen_query_para)
                gemma_para_correct = compare_sql(db_full_path, original_sql, gemma_query_para)

                # Store results
                original_result = {
                    "llama_original_correct": llama_original_correct,
                    "qwen_original_correct": qwen_original_correct,
                    "gemma_original_correct": gemma_original_correct
                }
                paraphrased_result = {
                    "llama_para_correct": llama_para_correct,
                    "qwen_para_correct": qwen_para_correct,
                    "gemma_para_correct": gemma_para_correct
                }

                for col, val in original_result.items():
                    paraphrased_df.loc[i, col] = val
                for col, val in paraphrased_result.items():
                    paraphrased_df.loc[i, col] = val

                all_correct = all(original_result.values()) and all(paraphrased_result.values())
                paraphrased_df.loc[i, "all_models_correct"] = int(all_correct)

                logger.info(f"[Row {i}] Original SQL correctness: {original_result}")
                logger.info(f"[Row {i}] Paraphrased SQL correctness: {paraphrased_result}")

            except Exception as e:
                logger.error(f"[Row {i}] NL2SQL error: {e}")

        paraphrased_df.to_csv(result_path, index=False)
        logger.info(f"NL2SQL evaluation complete. Results saved to: {result_path}")
    else:
        logger.info("Skipping NL2SQL generation.")

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    LOG_PATH = Path().resolve() / "logs" / "main.log"
    logger = setup_logger("main_logger", log_file=LOG_PATH)
    main()
