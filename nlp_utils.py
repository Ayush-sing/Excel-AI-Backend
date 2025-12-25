# nlp_utils.py
import re
import difflib
from difflib import SequenceMatcher
import pandas as pd

def extract_column_and_condition(message, df=None):
    """
    Extract target column and condition from natural language message.
    - Fuzzy match for column names (returns one column or None)
    - Dynamic detection of conditions from dataset values (supports numeric comparisons and text equality)
    Returns: (column_name_or_None, condition_string_or_None)
    Example condition string: "Region == 'East'" or "profit > 500"
    """
    column = None
    condition = None
    message_lower = (message or "").lower()

    # -------- Column detection (fuzzy) --------
    if df is not None:
        best_match = None
        best_score = 0
        for col in df.columns:
            col_lower = str(col).lower()
            overlap = sum(1 for word in col_lower.split() if word in message_lower)
            
            # handle partial/fuzzy matches
            if col_lower in message_lower or any(col_word[:-1] in message_lower for col_word in col_lower.split()):
                overlap += 1

            if overlap > best_score:
                best_match = col
                best_score = overlap
        if best_match:
            column = best_match

    # -------- Condition detection (text + numeric + comparison) --------
    if df is not None:
        for col in df.columns:
            col_low = str(col).lower()

            # Numeric comparisons: >, <, =  (matches numbers with optional decimal)
            match_gt = re.search(rf"{col_low}\s*(?:>|greater than|more than|above)\s*([0-9]+(?:\.[0-9]+)?)", message_lower)
            match_lt = re.search(rf"{col_low}\s*(?:<|less than|below|under)\s*([0-9]+(?:\.[0-9]+)?)", message_lower)
            match_eq_num = re.search(rf"{col_low}\s*(?:=|is|equals?|equal to)\s*([0-9]+(?:\.[0-9]+)?)", message_lower)

            if match_gt:
                condition = f"{col} > {match_gt.group(1)}"
                break
            if match_lt:
                condition = f"{col} < {match_lt.group(1)}"
                break
            if match_eq_num:
                condition = f"{col} == {match_eq_num.group(1)}"
                break

            # Text/categorical equality: check cell values (case-insensitive substring match)
            if df[col].dtype == object or df[col].dtype.name == "category":
                for val in pd.unique(df[col].dropna().astype(str)):
                    val_str = str(val).lower()
                    if val_str in message_lower:
                        # quote the value to preserve spaces/special chars
                        condition = f"{col} == '{val}'"
                        break
            if condition:
                break

    # -------- Regex fallback (captures "for X" type clauses) --------
    if condition is None:
        m = re.search(r"for\s+([^\.;,]+)", message_lower)
        if m:
            # return the raw clause as fallback (caller may parse further)
            condition = m.group(1).strip()

    return column, condition


# --- Helper wrapper: return up to 2 columns for use in chart/regression logic ---

def extract_columns_from_message(message, df, max_cols: int = 2):
    """
    Extract columns strictly based on {column name} syntax.
    If max_cols is None → return all detected columns (used for multiple regression)
    """
    if not message or df is None:
        return []

    msg_low = message.lower()
    brace_tokens = re.findall(r"\{([^}]+)\}", msg_low)
    if not brace_tokens:
        return []

    df_cols_lower = {str(c).lower(): c for c in df.columns}
    found = []

    for token in brace_tokens:
        tok = token.strip().lower()
        if tok in df_cols_lower:
            found.append(df_cols_lower[tok])
        else:
            close = difflib.get_close_matches(tok, list(df_cols_lower.keys()), n=1, cutoff=0.9)
            if close:
                found.append(df_cols_lower[close[0]])

    # remove duplicates, keep order
    seen = set()
    clean = []
    for c in found:
        if c not in seen:
            seen.add(c)
            clean.append(c)

    # 🔑 KEY CHANGE
    if max_cols is None:
        return clean

    return clean[:max_cols]






# --- Helper wrapper: return condition as dict {column: (op, value)} or {} if no condition found ---
def extract_condition_from_message(message, df):
    """
    Parse a single WHERE condition from the message.
    Supports brace columns:  where {Order ID} = 123
    And plain:               where Order ID = 123
    Returns dict: { matched_df_column: (op, value) }  or {}
    """
    if message is None:
        return {}
    text = message.strip()
    if " where " not in text.lower():
        return {}

    # take the part after 'where'
    cond_part = re.split(r"\bwhere\b", text, flags=re.IGNORECASE, maxsplit=1)[-1].strip()

    # 1) Brace style: {Col} <op> value
    m = re.search(r"\{([^}]+)\}\s*(=|==|is|>|<|>=|<=)\s*([^\n\r]+)$", cond_part, flags=re.IGNORECASE)
    if m:
        raw_col = m.group(1).strip()
        op = m.group(2).strip()
        val = m.group(3).strip()
        # normalize operator
        op = "=" if op.lower() in ("=", "==", "is") else op
        # match to df column
        for c in df.columns:
            if str(c).lower() == raw_col.lower():
                return {c: (op, val)}
        # small fuzzy if brace col off by a bit
        cols_lower = [str(c).lower() for c in df.columns]
        close = difflib.get_close_matches(raw_col.lower(), cols_lower, n=1, cutoff=0.82)
        if close:
            c = df.columns[cols_lower.index(close[0])]
            return {c: (op, val)}
        return {}

    # 2) Plain: Col <op> value   (kept only for compatibility if someone forgets braces)
    m2 = re.search(r"(.+?)\s*(=|==|is|>|<|>=|<=)\s*([^\n\r]+)$", cond_part, flags=re.IGNORECASE)
    if m2:
        raw_col = m2.group(1).strip()
        op = m2.group(2).strip()
        val = m2.group(3).strip()
        op = "=" if op.lower() in ("=", "==", "is") else op
        # case-insensitive exact match first
        for c in df.columns:
            if str(c).lower() == raw_col.lower():
                return {c: (op, val)}
        # fuzzy fallback
        cols_lower = [str(c).lower() for c in df.columns]
        close = difflib.get_close_matches(raw_col.lower(), cols_lower, n=1, cutoff=0.88)
        if close:
            c = df.columns[cols_lower.index(close[0])]
            return {c: (op, val)}
    return {}