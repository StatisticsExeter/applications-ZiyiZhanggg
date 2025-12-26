from pathlib import Path
from dotenv import load_dotenv
import os
try:
    import psycopg2
except ModuleNotFoundError:
    psycopg2 = None
import pandas as pd
import sys
import textwrap
try:
    from doit.exceptions import TaskFailed
except ModuleNotFoundError:
    TaskFailed = RuntimeError



def warn_if_in_rstudio():
    """Detect if running under RStudio/reticulate and print a warning."""
    # reticulate embeds Python in-process and sets these environment variables
    in_rstudio = any(
        "RSTUDIO" in k or "RETICULATE" in k for k in os.environ.keys()
    )

    if in_rstudio:
        msg = textwrap.dedent(f"""
        WARNING: This script appears to be running inside RStudio (reticulate).
        RStudio embeds Python in the R process, which can cause segfaults with
        libraries like psycopg2 and NumPy.

        Please re-run this script from a clean Anaconda prompt instead:
          conda activate python-exercises
          python {os.path.basename(sys.argv[0])}
        """)
        print(msg, file=sys.stderr)


def check_input_file(path, hint=None):
    """Return a TaskFailed object (not raise it) if the input file is missing."""
    if not os.path.exists(path):
        msg = f"Missing required input file: {path}"
        if hint:
            msg += f"\n Hint: {hint}"
        # Return TaskFailed so doit interprets it as a controlled failure
        print(msg, file=sys.stderr, flush=True)
        return TaskFailed(msg)
    return True


def load_pg_data(query):
    config = load_db_config()
    with get_db_connection(config) as conn:
        df = fetch_pg_data(conn, query)
    return df


def find_project_root(marker=".git"):
    """
    Traverse upward from the current file to find the project root.
    Includes the current file's directory in the search.
    """
    current_dir = Path.cwd().resolve()
    for directory in [current_dir] + list(current_dir.parents):
        if (directory / marker).exists():
            return directory
    raise RuntimeError("Project root not found. Please ensure a marker file exists.")


def load_db_config(env_path=None):
    if env_path is None:
        env_path = find_project_root()
    load_dotenv(dotenv_path=env_path / ".Renviron", override=True)
    return {
        "dbname": os.getenv("PGRDATABASE"),
        "user": os.getenv("PGRUSER"),
        "password": os.getenv("PGRPASSWORD"),
        "host": os.getenv("PGRHOST"),
        "port": os.getenv("PGRPORT"),
    }


def get_db_connection(config):
    if psycopg2 is None:
        raise ModuleNotFoundError(
            "psycopg2 is not installed in this environment, so database access is unavailable."
        )
    return psycopg2.connect(**config)


def fetch_pg_data(conn, query):
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=colnames)
