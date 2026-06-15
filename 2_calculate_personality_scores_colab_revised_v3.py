"""
calculate_personality_scores_colab.py
====================================

This script converts a stats‑ready long‑format CSV (produced by
``clean_long_colab.py``) into a wide‑format dataset with one row per
participant.  It computes personality trait scores from the Mini‑IPIP
(Big Five), Marlowe–Crowne Social Desirability Scale (MCSDS) adapted to
a five‑point Likert format, and the Short Dark Triad (SD3).  It also
retains demographic variables (gender, age) and any free‑text responses
(essay questions and feedback).

Usage in a Colab notebook:

```python
import calculate_personality_scores_colab as cps
cps.process_scores('stats_ready_long.csv', 'personality_scores_FORMAL.csv')
```

The function ``process_scores`` will read the long‑format file, pivot it
to wide format, compute trait scores, and save the result.  You can
override the input and output file names by passing different paths.

Scoring details:

* **Big Five (Mini‑IPIP)** – Items are coded as ``1E``, ``2A``, … ``20I``.
  The script maps verbal responses (``"Disagree Strongly"`` to ``"Agree Strongly"``)
  to numeric values 1–5.  Eleven items are reverse‑scored (6E, 7A, 8C, 9N,
  10I, 15I, 16E, 17A, 18C, 19N, 20I) as indicated in the Mini-IPIP scoring key. Note that the original Mini-IPIP uses accuracy anchors, whereas this experiment administered the items with agree/disagree anchors to match the other questionnaire scales; the numeric 1–5 scoring is unchanged.
  Trait scores are the mean of the four relevant items for Extraversion,
  Agreeableness, Conscientiousness, Neuroticism and Intellect/Imagination.

* **Social Desirability (MCSDS)** – Items ``SDS1`` through ``SDS33`` are
  treated on a five‑point scale.  A key vector specifies which items
  represent socially desirable “True” responses; those that are keyed
  ``False`` are reverse‑scored (5 → 1, 4 → 2, etc.).  The score is the
  participant’s mean across the 33 items.  Thresholds for low, average
  and high social desirability (0–8, 9–19 and 20–33 on the binary scale)
  come from Crowne and Marlowe’s original binary-score categories,
  though this adapted five-point version reports only the continuous mean.

* **Short Dark Triad (SD3)** – Items are labelled ``sd3m1`` … ``sd3m9``
  (Machiavellianism), ``sd3n1`` … ``sd3n9`` (Narcissism) and ``sd3p1`` …
  ``sd3p9`` (Psychopathy).  Five items are reverse‑scored:
  ``sd3n2``, ``sd3n6``, ``sd3n8``, ``sd3p2`` and ``sd3p7``.  Subscale
  scores are the mean across the nine items for each domain.  Normative
  cut‑offs (Machiavellianism > 3.86, Narcissism > 3.68, Psychopathy > 3.40) can
  be used to flag high scores, but classification is left to the analyst.

The script also automatically includes any other columns from the long
data (e.g., essay questions or feedback) by pivoting them into the
wide dataset.
"""

import re
import hashlib
import pandas as pd
from typing import Dict, List, Optional

# Mapping from verbal Likert responses to numeric scores
# Likert response mapping for verbal responses.
# These values correspond to a five‑point agreement scale.  When scoring
# questionnaire items we convert text (e.g., ``"Agree"``) into its
# numeric counterpart.  Numeric strings (``"4"`` or ``"2.0"``) are
# also permitted and will be coerced to floats.  Any unrecognised
# responses (including empty strings or free‑text answers) are
# treated as missing (NaN).
LIKERT_MAP: Dict[str, int] = {
    'Disagree Strongly': 1,
    'Disagree': 2,
    'Neither Agree nor Disagree': 3,
    'Agree': 4,
    'Agree Strongly': 5
}


VALID_CONFIRMATION_RE = r'^[A-Z0-9]{8}$'

def valid_confirmation_code(x) -> bool:
    """Return True only for real completion codes: 8 uppercase alphanumeric characters."""
    if pd.isna(x):
        return False
    return bool(re.fullmatch(VALID_CONFIRMATION_RE, str(x).strip()))

def make_participant_id(row) -> str:
    """Create the same hashed participant_id used by the LLM essay-rating script."""
    raw = "|".join([
        str(row.get('run_id', '')),
        str(row.get('recorded_at', '')),
        str(row.get('ip', '')),
        str(row.get('user_agent', '')),
    ])
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]

def infer_target_trait(question_id: str) -> Optional[str]:
    q = str(question_id)
    if q.startswith('machiv_clean'):
        return 'Machiavellianism'
    if q.startswith('npi_clean'):
        return 'Narcissism'
    if q.startswith('srp_clean'):
        return 'Psychopathy'
    return None

def infer_prompt_type(question_id: str) -> Optional[str]:
    q = str(question_id).lower()
    if q.endswith('_broad'):
        return 'broad'
    if q.endswith('_specific'):
        return 'specific'
    return None

def get_formal_complete_participant_ids(df_long: pd.DataFrame) -> List[str]:
    """Return participant_ids with valid confirmation code and exactly six balanced valid essays."""
    if 'participant_id' not in df_long.columns:
        df_long['participant_id'] = df_long.apply(make_participant_id, axis=1)
    df_long['valid_confirmation_code'] = df_long['confirmationCode'].apply(valid_confirmation_code)
    confirmed_ids = set(df_long.loc[df_long['valid_confirmation_code'], 'participant_id'].dropna().astype(str))

    essay_mask = df_long['question_id'].astype(str).str.startswith(('machiv_clean', 'npi_clean', 'srp_clean'))
    essays = df_long[df_long['participant_id'].astype(str).isin(confirmed_ids) & essay_mask].copy()
    if 'valid' in essays.columns:
        essays = essays[~essays['valid'].astype(str).str.lower().eq('false')].copy()
    essays = essays[essays['value'].notna() & (essays['value'].astype(str).str.strip() != '')].copy()
    essays['target_trait'] = essays['question_id'].apply(infer_target_trait)
    essays['prompt_type'] = essays['question_id'].apply(infer_prompt_type)
    if 'trial_index' in essays.columns:
        essays['trial_index_num'] = pd.to_numeric(essays['trial_index'], errors='coerce')
        essays = essays.sort_values(['participant_id', 'question_id', 'trial_index_num'])
    essays = essays.drop_duplicates(['participant_id', 'question_id'], keep='last')

    keep_ids = []
    for pid, group in essays.groupby('participant_id'):
        if len(group) != 6:
            continue
        trait_counts = group['target_trait'].value_counts().to_dict()
        if trait_counts.get('Machiavellianism', 0) != 2 or trait_counts.get('Narcissism', 0) != 2 or trait_counts.get('Psychopathy', 0) != 2:
            continue
        ok = True
        for trait in ['Machiavellianism', 'Narcissism', 'Psychopathy']:
            pt = group[group['target_trait'] == trait]['prompt_type'].value_counts().to_dict()
            if pt.get('broad', 0) != 1 or pt.get('specific', 0) != 1:
                ok = False
                break
        if ok:
            keep_ids.append(str(pid))
    return keep_ids


def map_likert(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of Likert responses to numeric values.

    This helper performs three steps for each element in ``series``:

      1. If the value is already numeric (int/float), return it as a float.
      2. If the value matches a key in ``LIKERT_MAP`` (e.g., "Agree"),
         return the mapped integer as a float.
      3. If the value looks like a numeric string (e.g., "4" or "3.0"),
         attempt to coerce it to a float.
      4. Otherwise return NaN.

    This approach prevents TypeError when performing arithmetic on
    questionnaire responses because all recognised values are converted
    to floats.  Unknown values (including free‑text responses) will
    propagate as NaN and be ignored in means.
    """
    def convert(value):
        # Handle missing values explicitly
        if pd.isna(value):
            return float('nan')
        # If value is already numeric, return as float
        if isinstance(value, (int, float)):
            return float(value)
        # Strip whitespace and normalise text
        text = str(value).strip()
        # Check if text matches a mapped Likert response
        if text in LIKERT_MAP:
            return float(LIKERT_MAP[text])
        # Try to coerce numeric strings (e.g. "4", "2.0") to float
        try:
            return float(text)
        except ValueError:
            # Unrecognised values become NaN
            return float('nan')
    return series.apply(convert)


def compute_big5(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Big Five trait scores from Mini‑IPIP items.

    Adds five new columns to the DataFrame: Extraversion, Agreeableness,
    Conscientiousness, Neuroticism and Intellect.
    """
    traits: Dict[str, List[str]] = {
        'Extraversion':      ['1E', '6E', '11E', '16E'],
        'Agreeableness':     ['2A', '7A', '12A', '17A'],
        'Conscientiousness': ['3C', '8C', '13C', '18C'],
        'Neuroticism':       ['4N', '9N', '14N', '19N'],
        'Intellect':         ['5I', '10I', '15I', '20I']
    }
    reverse_items: set = {
        '6E', '7A', '8C', '9N', '10I', '15I',
        '16E', '17A', '18C', '19N', '20I'
    }
    for trait, items in traits.items():
        # Be defensive: if an expected item column is absent, create it as missing.
        # This keeps the function consistent with compute_sds() and compute_sd3().
        for item in items:
            if item not in df.columns:
                df[item] = pd.NA
        numeric = df[items].apply(map_likert)
        for col in numeric.columns:
            if col in reverse_items:
                numeric[col] = numeric[col].apply(lambda x: 6 - x if pd.notna(x) else x)
        df[trait] = numeric.mean(axis=1, skipna=True)
    return df


def compute_sds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the MCSDS social desirability mean score.

    Adds one new column, `SDS_mean`, to the DataFrame.
    """
    sds_items: List[str] = [f'SDS{i}' for i in range(1, 34)]
    # Ensure all items exist in the DataFrame
    for item in sds_items:
        if item not in df.columns:
            df[item] = pd.NA
    numeric = df[sds_items].apply(map_likert)
    sds_key = [
        True, True, False, True, False, False, True, True, False, False,
        False, False, True, False, False, True, True, True, False, True,
        True, False, False, True, True, True, True, False, True, False,
        True, False, True
    ]
    for idx, keyed_true in enumerate(sds_key):
        if not keyed_true:
            col = sds_items[idx]
            numeric[col] = numeric[col].apply(lambda x: 6 - x if pd.notna(x) else x)
    df['SDS_mean'] = numeric.mean(axis=1, skipna=True)
    return df


def compute_sd3(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Short Dark Triad subscale scores.

    Adds three new columns: Machiavellianism, Narcissism and Psychopathy.
    """
    sd3_scales: Dict[str, List[str]] = {
        'Machiavellianism': [f'sd3m{i}' for i in range(1, 10)],
        'Narcissism':       [f'sd3n{i}' for i in range(1, 10)],
        'Psychopathy':      [f'sd3p{i}' for i in range(1, 10)]
    }
    reverse_items: set = {'sd3n2', 'sd3n6', 'sd3n8', 'sd3p2', 'sd3p7'}
    for trait, items in sd3_scales.items():
        for item in items:
            if item not in df.columns:
                df[item] = pd.NA
        numeric = df[items].apply(map_likert)
        for col in numeric.columns:
            if col in reverse_items:
                numeric[col] = numeric[col].apply(lambda x: 6 - x if pd.notna(x) else x)
        df[trait] = numeric.mean(axis=1, skipna=True)
    return df


def pivot_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format data into wide format using collision-safe participant_id.

    The participant_id is the same hashed ID used by the D3 LLM-rating script:
    hash(run_id | recorded_at | ip | user_agent)[:12]. The pivot no longer uses
    run_id/confirmationCode as the participant key, avoiding collisions across
    Cognition.run exports.
    """
    if 'participant_id' not in df_long.columns:
        df_long = df_long.copy()
        df_long['participant_id'] = df_long.apply(make_participant_id, axis=1)

    # Strictly validate completion codes and keep a clean code only when real.
    df_long['valid_confirmation_code'] = df_long['confirmationCode'].apply(valid_confirmation_code)
    df_long['confirmationCode_clean'] = df_long['confirmationCode'].where(df_long['valid_confirmation_code'], pd.NA)

    wide = df_long.pivot_table(
        index=['participant_id'],
        columns='question_id',
        values='value',
        # Use the last non-missing value for consistency with the essay-completeness
        # logic, which keeps the last duplicate by trial_index when duplicates occur.
        aggfunc=lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else pd.NA
    )
    wide.reset_index(inplace=True)

    # Add session-level metadata. Raw ip/user_agent are deliberately not retained.
    # Use the older pandas-compatible dict aggregation form rather than
    # named aggregation, because some local Anaconda/Python 3.7 installs
    # do not preserve the group key correctly with named aggregation.
    def first_nonmissing(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else pd.NA

    meta = (
        df_long.groupby('participant_id')
        .agg({
            'run_id': 'first',
            'recorded_at': 'first',
            'confirmationCode_clean': first_nonmissing,
            'valid_confirmation_code': 'max',
        })
        .reset_index()
        .rename(columns={
            'confirmationCode_clean': 'confirmationCode',
            'valid_confirmation_code': 'has_valid_confirmation_code',
        })
    )
    wide = wide.merge(meta, on='participant_id', how='left')

    if 'total_rt' in df_long.columns:
        total_rt = df_long.groupby('participant_id')['total_rt'].max().reset_index().rename(columns={'total_rt': 'total_rt'})
        wide = wide.merge(total_rt, on='participant_id', how='left')
    return wide

def process_scores(input_csv: str, output_csv: str) -> None:
    """
    Compute personality trait scores from a long‑format dataset and
    write a wide‑format summary file.

    The function performs the following steps:

      1. Reads a ``stats_ready_long`` CSV generated by ``clean_long_colab``.
      2. Pivots the long data so each participant is a row and each
         ``question_id`` is a column.
      3. Computes Big‑Five trait means, the social desirability mean,
         and Short Dark Triad subscale means, applying reverse scoring
         where necessary.
      4. Identifies demographic fields (``gender``, ``age``), essay and
         feedback items (columns containing ``machiv_clean``, ``npi_clean``,
         ``srp_clean`` or named ``feedback``), and appends them to the
         trait scores.
      5. Writes the resulting wide‑format DataFrame with one row per
         participant to ``output_csv``.

    Parameters
    ----------
    input_csv : str
        Path to the stats‑ready long CSV.
    output_csv : str
        Path where the wide‑format scores CSV should be written.
    """
    # 1. Read the long-format data
    df_long = pd.read_csv(input_csv)

    required_for_id = {'run_id', 'recorded_at', 'ip', 'user_agent', 'confirmationCode'}
    missing_for_id = required_for_id - set(df_long.columns)
    if missing_for_id:
        raise ValueError(f'Input missing columns required for collision-safe participant_id: {missing_for_id}')
    df_long['participant_id'] = df_long.apply(make_participant_id, axis=1)

    # Keep only formal experiment participants: valid 8-character confirmation code + exactly six balanced valid essays.
    keep_ids = get_formal_complete_participant_ids(df_long)
    df_long = df_long[df_long['participant_id'].astype(str).isin(keep_ids)].copy()
    print(f'Formal complete participants kept for scoring: {len(keep_ids)}')

    # 2. Pivot to wide format using participant_id, not run_id/confirmationCode.
    wide = pivot_long_to_wide(df_long)
    # 3. Compute trait scores
    wide = compute_big5(wide)
    wide = compute_sds(wide)
    wide = compute_sd3(wide)
    # 4. Select relevant columns
    demographic_cols = [c for c in ['gender', 'age'] if c in wide.columns]
    # Essay and feedback columns: identify any cleaned SD3 text responses or feedback field
    essay_cols = [c for c in wide.columns
                  if (c.startswith('machiv_clean') or c.startswith('npi_clean')
                      or c.startswith('srp_clean') or c.lower() == 'feedback')]
    # Build output columns list
    output_cols = (
        ['participant_id', 'run_id', 'recorded_at', 'confirmationCode', 'has_valid_confirmation_code', 'total_rt']
        + demographic_cols
        + ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Intellect',
           'SDS_mean', 'Machiavellianism', 'Narcissism', 'Psychopathy']
        + essay_cols
    )
    # Ensure unique columns and presence in DataFrame
    output_cols = [c for c in dict.fromkeys(output_cols) if c in wide.columns]
    # 5. Write the wide‑format DataFrame to disk
    wide[output_cols].to_csv(output_csv, index=False)
    print(f"Wrote personality scores to {output_csv}")


def assert_participant_id_match(rating_input_csv: str, scores_csv: str) -> None:
    """Assert that the LLM-rating input and score file contain the same participants.

    Use this after generating both files. It catches silent sample divergence
    before merging questionnaire scores with LLM ratings.
    """
    rating_input = pd.read_csv(rating_input_csv)
    scores = pd.read_csv(scores_csv)
    if 'participant_id' not in rating_input.columns:
        raise ValueError(f"{rating_input_csv} does not contain participant_id")
    if 'participant_id' not in scores.columns:
        raise ValueError(f"{scores_csv} does not contain participant_id")

    rating_ids = set(rating_input['participant_id'].dropna().astype(str))
    score_ids = set(scores['participant_id'].dropna().astype(str))
    if rating_ids != score_ids:
        only_rating = sorted(rating_ids - score_ids)
        only_scores = sorted(score_ids - rating_ids)
        raise AssertionError({
            'in_rating_not_scores': only_rating,
            'in_scores_not_rating': only_scores,
        })
    print(f"Participant ID sets match: N = {len(score_ids)}")


if __name__ == '__main__':
    process_scores('stats_ready_long.csv', 'personality_scores_FORMAL.csv')