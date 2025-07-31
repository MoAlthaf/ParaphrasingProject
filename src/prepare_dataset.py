from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
from utils.sql_utils import extract_schema , get_sample_rows ,run_query
import json

load_dotenv()

OPENAI_API_KEY=os.getenv("OPEN_AI")


client=OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH= Path(PROJECT_ROOT) / "data" / "database"   # Path to Spider database
OUTPUT_PATH=Path(PROJECT_ROOT) / "data" / "interim"  # Path to output directory


def generate_sql_query(schema,sample_rows=None):
    no_of_queries = 2  # Number of queries to generate
    system_prompt = (
    "You are an expert SQL query generator.\n"
    "Given a database schema and example rows, generate a list of SQL queries.\n"
    "Requirements:\n"
    "- Only generate syntactically correct SQL queries for a SQLite database.\n"
    "- Do not hallucinate tables or column names.\n"
    "- Use actual values seen in the sample rows.\n"
    "- Cover diverse query types (SELECT, JOIN, GROUP BY, ORDER BY, etc.).\n"
    "- Do not include natural language questions, comments, or any explanation.\n"
    "- Do not include markdown, text, or formatting.\n"
    "- Output ONLY a JSON array of SQL query strings.\n"
    "- If unsure about any logic, skip it — do not assume."
    )

    user_prompt = (
        f"Here is the database schema, Generate {no_of_queries} queries: \n"
        "Database Schema:\n"
        f"{schema} \n"
    
        "Please provide the SQL query."
        
        )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    queries=json.loads(response.choices[0].message.content)
    return queries

def generarte_nl(sql_query,result):
    
    system_prompt=(
        "You are a helpful assistant that generates natural language question for the given sql query and the result.\n"
        "- "
    )


if __name__=="__main__":

    for db_file in DB_PATH.iterdir():
        db_full_path= Path(DB_PATH) / db_file / f"{db_file.name}.sqlite"
        schema= extract_schema(db_full_path)
        schema_text=schema
        sample_rows=get_sample_rows(db_full_path)  # Replace "table_name" with an actual table name from your schema
        sql_queries=generate_sql_query(schema_text)
        for query in sql_queries:
            print("Here is the Query: ",query)
            result=run_query(db_full_path,query)
            if not result.empty:
                print(result.head())
            else:
                print("The Query is wrong:")
            print()
        break
    #print(sample_rows)