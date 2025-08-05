from pathlib import Path
import pandas as pd
from multiprocessing import set_start_method

from src.prepare_dataset import main as prepare_dataset
from models.llama import paraphrase_sentence, regenerate_paraphrase,generate_sql as nl2sql_llama
from src.utils.sql_utils import extract_schema
from src.utils.logger import setup_logger
from src.utils.paraphrase_score import score_paraphrase

def main():
    THRESHOLD = 0.7  # Set your acceptable paraphrasing quality threshold
    MAX_RETRIES = 1  # Avoid infinite regeneration

    PROJECT_ROOT = Path().resolve()
    
    # Paths
    dataset_path = PROJECT_ROOT / "data" / "interim" / "generated_queries.csv"
    paraphrased_csv_path = PROJECT_ROOT / "data" / "processed" / "output_paraphrased.csv"
    database_path = PROJECT_ROOT / "data" / "database"

    # Force flags
    dataset_force = False   #Change these values to regenerate dataset even if it exists.
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
                score= score_paraphrase(paraphrased=paraphrased,original= question)

                if score < THRESHOLD:
                    logger.warning(f"[Row {i}] Low paraphrasing score ({score:.2f}). Retrying...")
                    for attempt in range(MAX_RETRIES):
                        paraphrased = regenerate_paraphrase(question, schema)
                        score = score_paraphrase(question, paraphrased)
                        logger.info(f"[Row {i}] Retry score: {score:.2f}")
                        if score >= THRESHOLD:
                            break
            except Exception as e:
                logger.warning(f"[Row {i}] Paraphrasing error: {e}")
                paraphrased = question
                score=0.0

            dataset_df.loc[i, "paraphrased_nl"] = paraphrased
            dataset_df.loc[i, "paraphrased_score"] = score

            logger.info(f"[Row {i}] Original: {question}")
            logger.info(f"[Row {i}] Paraphrased: {paraphrased}")
            logger.info(f"[Row {i}] Score: {score:.2f}")

        dataset_df.to_csv(paraphrased_csv_path, index=False)
        logger.info(f"Paraphrasing complete. Output saved to: {paraphrased_csv_path}")
    else:
        logger.info("Paraphrased dataset already exists. Skipping...")

    # Step 3: NL2SQL (placeholder)
    if nl2sql_force:
        logger.info("Starting NL2SQL generation...")
        # todo: add NL2SQL logic here
        pass
    else:
        logger.info("Skipping NL2SQL generation.")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    LOG_PATH = Path().resolve() / "logs" / "main.log"
    logger = setup_logger("main_logger", log_file=LOG_PATH)
    main()
