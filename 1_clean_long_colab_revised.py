# (See module docstring below for usage instructions)

import json
import hashlib
import pandas as pd
from typing import List

"""
This module was originally designed to be called from the command line with
`--input` and `--output` arguments.  To make it easier for programming
beginners to use within a Colab notebook, the argument parser has been
removed and default file names are defined at the top of the file.  You
can modify the `INPUT_CSV` and `OUTPUT_CSV` variables below to point to
your own data file and desired output name.  Then simply run

```python
import clean_long_colab
clean_long_colab.process_file()
```

This will produce a long‑format CSV in your working directory with one
row per participant per question, including the participant’s total
reaction time. Participant sessions are identified with a collision-safe
``participant_id`` hash rather than ``run_id`` alone, because ``run_id`` may
collide across Cognition.run exports.  Should you wish to call this script from the command
line, you can still do so by setting the `INPUT_CSV` and `OUTPUT_CSV`
environment variables before execution.
"""

# Set these values to your input and output file names
INPUT_CSV = 'llmpsytrial.csv'
OUTPUT_CSV = 'stats_ready_long.csv'


def make_participant_id(row) -> str:
    """Create a collision-safe participant_id from session-level metadata.

    This uses the same hash construction as the LLM essay-rating and
    personality-scoring scripts: SHA-256(run_id|recorded_at|ip|user_agent),
    truncated to the first 12 hexadecimal characters. Raw ip/user_agent are
    retained in the intermediate long file for audit and reproducibility, but
    downstream analysis files should avoid exposing them.
    """
    raw = "|".join([
        str(row.get('run_id', '')),
        str(row.get('recorded_at', '')),
        str(row.get('ip', '')),
        str(row.get('user_agent', '')),
    ])
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]


def load_and_trim(input_csv: str, keep_cols: List[str]) -> pd.DataFrame:
    """Load the CSV file and keep only specified columns.

    Parameters
    ----------
    input_csv : str
        Path to the raw jsPsych CSV file.
    keep_cols : List[str]
        Columns to retain from the input file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the specified columns.
    """
    # Using low_memory=False to suppress mixed type warnings
    df = pd.read_csv(input_csv, usecols=lambda c: c in keep_cols, low_memory=False)
    # Drop exact duplicate rows to avoid double processing
    df = df.drop_duplicates()
    return df


def compute_total_rt(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the total reaction time per participant.

    The `rt` column in jsPsych exports represents the reaction time for a
    single trial in milliseconds.  To estimate the total time spent on the
    survey, the script sums these values across all trials for each
    participant (identified by collision-safe `participant_id`). Non‑numeric
    values are coerced to NaN and ignored in the sum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least `participant_id` and `rt` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `participant_id` and `total_rt`.
    """
    # Convert rt to numeric (milliseconds).  Non‑numeric values become NaN.
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    # Sum rt by participant_id, not run_id. run_id can collide across exports.
    total_rt = (df.groupby('participant_id')['rt']
                  .sum(min_count=1)
                  .reset_index()
                  .rename(columns={'rt': 'total_rt'}))
    return total_rt


def parse_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the JSON responses into a long‑format DataFrame.

    Each row in the input `df` corresponds to a trial.  The `response`
    column may contain a JSON‑encoded string with one or more
    questionnaire responses.  This function iterates over rows,
    parses valid JSON strings, and expands key–value pairs into
    individual records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing `participant_id`, `run_id`, `confirmationCode`, and `response` columns.

    Returns
    -------
    pd.DataFrame
        A long‑format DataFrame with `participant_id`, `run_id`,
        `confirmationCode`, `question_id`, and `value` columns.
    """
    records = []
    for _, row in df.iterrows():
        participant_id = row.get('participant_id')
        run_id = row.get('run_id')
        confirm = row.get('confirmationCode')
        resp_str = row.get('response')
        valid = row.get('valid')
        trial_index = row.get('trial_index')
        trial_type = row.get('trial_type')
        ip = row.get('ip')
        recorded_at = row.get('recorded_at')
        user_agent = row.get('user_agent')
        if not isinstance(resp_str, str):
            continue
        # Only attempt to parse if it looks like a JSON object
        resp_str = resp_str.strip()
        if len(resp_str) > 1 and resp_str.startswith('{') and resp_str.endswith('}'):
            try:
                resp_dict = json.loads(resp_str)
            except json.JSONDecodeError:
                # Skip badly formatted JSON
                continue
            for q_id, value in resp_dict.items():
                records.append({
                    'participant_id': participant_id,
                    'run_id': run_id,
                    'confirmationCode': confirm,
                    'question_id': q_id,
                    'value': value,
                    'valid': valid,
                    'trial_index': trial_index,
                    'trial_type': trial_type,
                    'ip': ip,
                    'recorded_at': recorded_at,
                    'user_agent': user_agent
                })
    long_df = pd.DataFrame.from_records(records)
    return long_df


def process_file(input_csv: str = None, output_csv: str = None) -> None:
    """High‑level function to clean and convert a raw jsPsych CSV to long format.

    Parameters
    ----------
    input_csv : str, optional
        Path to the raw jsPsych CSV.  If None, uses the module‑level
        ``INPUT_CSV`` constant.
    output_csv : str, optional
        Path to write the long‑format CSV.  If None, uses the module‑level
        ``OUTPUT_CSV`` constant.

    Returns
    -------
    None
    """
    # Use defaults when arguments are not provided
    input_path = input_csv if input_csv is not None else INPUT_CSV
    output_path = output_csv if output_csv is not None else OUTPUT_CSV

    # Columns to retain from the raw export.  Additional columns can be
    # added here if needed for downstream analyses.
    keep_cols = [
        'run_id',
        'confirmationCode',
        'response',
        'rt',
        'valid',
        'trial_index',
        'trial_type',
        'ip',
        'recorded_at',
        'user_agent'
    ]
    print(f"Loading data from {input_path}…")
    df = load_and_trim(input_path, keep_cols)
    print(f"Initial rows after trimming: {len(df)}")

    required_for_id = {'run_id', 'recorded_at', 'ip', 'user_agent'}
    missing_for_id = required_for_id - set(df.columns)
    if missing_for_id:
        raise ValueError(f"Input missing columns required for collision-safe participant_id: {missing_for_id}")
    df['participant_id'] = df.apply(make_participant_id, axis=1)

    # Compute total reaction time per participant
    total_rt_df = compute_total_rt(df)
    print(f"Computed total_rt for {len(total_rt_df)} participants.")

    # Parse the responses into long format
    long_df = parse_responses(df)
    print(f"Parsed {len(long_df)} questionnaire responses before filtering invalid open-text attempts.")

    # Remove only explicitly invalid open-ended response attempts.
    # This keeps NaN/null valid values for demographics, closed questionnaires, feedback, etc.
    if 'valid' in long_df.columns:
        invalid_mask = long_df['valid'].astype(str).str.lower().eq('false')
        long_df = long_df[~invalid_mask].copy()

    print(f"Retained {len(long_df)} questionnaire responses after removing valid == False rows.")

    # Merge total_rt onto the long format data
    long_df = long_df.merge(total_rt_df, on='participant_id', how='left')

    # Drop rows with missing question_id or value (shouldn't occur)
    long_df = long_df.dropna(subset=['question_id', 'value'])

    # Reorder columns for clarity
    columns_order = [
        'participant_id',
        'run_id',
        'recorded_at',
        'confirmationCode',
        'question_id',
        'value',
        'total_rt',
        'valid',
        'trial_index',
        'trial_type',
        'ip',
        'user_agent'
    ]
    extra_cols = [c for c in long_df.columns if c not in columns_order]
    long_df = long_df[columns_order + extra_cols]

    # Write output
    print(f"Writing long‑format data to {output_path}…")
    long_df.to_csv(output_path, index=False)
    print("Done.")


# If this module is run directly (e.g. via %run in a notebook), call
# ``process_file`` using the default file names.  When imported, users can
# call ``process_file(input_csv, output_csv)`` with their own paths.
if __name__ == '__main__':
    process_file()