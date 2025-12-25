# (Full file content — updated create_chart_image to remove local imports.)
# ——————————————————————————————————————————————————————————————
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import io
import base64
import matplotlib
matplotlib.use("Agg")  # ✅ Use non-GUI backend for safe image rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import difflib
from difflib import SequenceMatcher
import re
from nlp_utils import extract_columns_from_message
from datetime import datetime
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

def _parse_date_flex(val: str):
    """
    Try parsing a user-supplied date string with dayfirst=True then False.
    Returns a normalized Timestamp (midnight) or NaT.
    """
    if val is None:
        return pd.NaT
    s = str(val).strip()
    # Try explicit common patterns first (fast path)
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(datetime.strptime(s, fmt)).normalize()
        except Exception:
            pass
    # Fallback: pandas parser both ways
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.normalize() if pd.notna(dt) else pd.NaT


def _series_to_datetime_flex(series: pd.Series) -> pd.Series:
    """
    Convert date-like column to datetime robustly.
    - If numeric (Excel serial days), use origin=1899-12-30.
    - Else try dayfirst=True then False.
    - Always normalize to midnight.
    """
    s = series

    # 1) Excel serial dates (numeric)
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        # Typical Excel serial day range ~ (20000 ~ 60000) for years 1954–2064
        plausibly_excel = s_num.between(20000, 60000, inclusive="both")
        if plausibly_excel.any():
            dt = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
            return dt

        # If numeric but not plausible Excel days, fall through to generic parse
        # (treat as text below)

    # 2) Text/objects — try dayfirst=True then False
    s_str = s.astype(str).str.strip()
    dt = pd.to_datetime(s_str, errors="coerce", dayfirst=True).dt.normalize()
    if dt.notna().sum() < max(3, int(0.5 * len(s_str))):
        dt = pd.to_datetime(s_str, errors="coerce", dayfirst=False).dt.normalize()

    return dt



def normalize_condition_value(df, col_name, val):
    """
    Normalize condition value to match column data type.
    - Fixes date format mismatch (27-01-2011 vs 2011-01-27)
    - Converts numeric strings to floats
    - Cleans categorical/text values
    """
    import pandas as pd
    import numpy as np

    if col_name not in df.columns:
        return val

    series = df[col_name]
    dtype = series.dtype

    # Handle datetime columns
    if np.issubdtype(dtype, np.datetime64):
        val_dt = pd.to_datetime(str(val), errors="coerce", dayfirst=True)
        if pd.notna(val_dt):
            return val_dt.normalize()
        return val

    # Handle numeric columns
    if pd.api.types.is_numeric_dtype(dtype):
        try:
            return float(val)
        except Exception:
            return val

    # Handle text/categorical
    return str(val).strip().lower()


def safe_numeric_stat(df, col, func_name):
    """
    Safely compute numeric or text-based statistics.
    Handles mixed/empty data and returns readable message.
    """
    try:
        func_name = func_name.lower()

        # --- handle mode separately for both text & numeric ---
        if func_name in ["mode"]:
            modes = df[col].mode()
            if modes.empty:
                return f"Mode of {col}: None"
            uniq_modes = sorted(set(modes.astype(str).tolist()))
            return f"Mode of {col} = {', '.join(uniq_modes)}"

        # --- numeric operations below ---
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            return f"⚠️ Column {col} has no numeric values to compute {func_name}."

        if func_name in ["sum", "total"]:
            val = series.sum()
        elif func_name in ["average", "mean"]:
            val = series.mean()
        elif func_name in ["median"]:
            val = series.median()
        elif func_name in ["variance", "var"]:
            val = series.var()
        elif func_name in ["std", "stddev", "standard deviation"]:
            val = series.std()
        elif func_name in ["min", "minimum"]:
            val = series.min()
        elif func_name in ["max", "maximum"]:
            val = series.max()
        elif func_name in ["range", "difference"]:
            val = series.max() - series.min()
        else:
            return f"⚠️ Statistic '{func_name}' not supported."

        if pd.isna(val):
            return f"⚠️ Could not compute {func_name} for {col} (insufficient data)."

        return f"✅ {func_name.title()} of {col} = {val}"

    except Exception as e:
        return f"❌ Error computing {func_name} of {col}: {str(e)}"



# ---------------- Helpers ---------------- #

def _to_img_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return data


def detect_chart_type(message: str) -> str:
    """
    Detect the most appropriate chart type from a user message.
    Includes support for advanced chart types while preserving existing heuristics.
    """

    msg = (message or "").lower().strip()

    # --- Priority detection for advanced charts ---
    if any(k in msg for k in ["heatmap", "correlation matrix", "matrix view", "correlation heatmap"]):
        return "heatmap"

    if any(k in msg for k in ["scatter matrix", "pairplot", "pair plot", "relationship matrix"]):
        return "scatter_matrix"

    if "pie" in msg and "chart" in msg:
        return "pie"

    if any(k in msg for k in ["hist", "histogram", "distribution", "frequency"]):
        return "hist"


    if any(k in msg for k in ["scatter", "relationship", "relation", "compare", "vs", "against"]):
        return "scatter"

    if any(k in msg for k in ["box", "boxplot", "spread", "iqr"]):
        return "box"

    # --- Your original fallbacks and safety heuristics ---
    if any(k in msg for k in ["bar", "column", "categorical", "count"]):
        return "bar"

    if any(k in msg for k in ["line","trend", "trendline"]):
        return "line"

    if any(k in msg for k in ["area", "fill chart", "stacked area"]):
        return "area"

    if "time" in msg or "date" in msg:
        return "time"
    
    # --- Broad chart keywords ---
    if "chart" in msg or "plot" in msg or "graph" in msg:
        # Try to infer intent by context
        if "time" in msg or "date" in msg or "month" in msg or "year" in msg:
            return "time"
        elif "category" in msg or "region" in msg or "type" in msg:
            return "bar"
        elif "share" in msg or "portion" in msg or "percentage" in msg:
            return "pie"
        elif "distribution" in msg:
            return "hist"
        elif "compare" in msg or "vs" in msg or "against" in msg:
            return "scatter"
        elif "heatmap" in msg:
            return "heatmap"
        else:
            return "bar"

    # --- Default ---
    return "bar"




def create_pie_chart(df, x_col, y_col=None):
    """
    Creates a readable pie/donut chart.
    - If y_col numeric -> sums y by x_col and plots share.
    - Otherwise -> plots counts of x_col.
    - Groups small slices into 'Other' (top 10 kept).
    """
     # --- 🧠 Auto-fix swapped arguments ---
    # Case 1: both columns provided and seem swapped
    if y_col and y_col in df.columns:
        if pd.api.types.is_numeric_dtype(df.get(x_col, pd.Series(dtype=float))) and not pd.api.types.is_numeric_dtype(df[y_col]):
            # x is numeric, y is categorical → swap
            x_col, y_col = y_col, x_col
        elif not pd.api.types.is_numeric_dtype(df.get(x_col, pd.Series(dtype=float))) and not pd.api.types.is_numeric_dtype(df[y_col]):
            # both non-numeric → keep original (counts mode)
            y_col = None
    else:
        y_col = None  # fallback if y_col missing or invalid



    # Defensive: ensure x_col exists
    if x_col not in df.columns:
        raise ValueError(f"x_col '{x_col}' not found in dataframe")

    # Prepare data series (index=category, values=numeric)
    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        # group sums by category, include NaN group with label 'Unknown'
        grouped = df.groupby(x_col, dropna=False)[y_col].sum()
        # move NaN index to string "Unknown" for nicer labels
        if grouped.index.hasnans:
            grouped = grouped.rename(index={np.nan: "Unknown"})
        # ensure numeric dtype
        grouped = pd.to_numeric(grouped, errors="coerce").fillna(0).sort_values(ascending=False)
    else:
        # categorical counts (including NaN)
        counts = df[x_col].astype(str).fillna("Unknown")
        grouped = counts.value_counts()
        grouped = pd.to_numeric(grouped, errors="coerce").fillna(0).sort_values(ascending=False)

    # If no data, show message
    if grouped.sum() == 0 or len(grouped) == 0:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No data to plot", ha="center")
        return plt.gcf()

    # Limit slices: keep top_n, rest -> "Other"
    top_n = 8
    if len(grouped) > top_n:
        top = grouped.iloc[:top_n]
        other_sum = grouped.iloc[top_n:].sum()
        grouped = pd.concat([top, pd.Series({"Other": other_sum})])
        # ensure we have consistent numeric dtype
        grouped = pd.to_numeric(grouped, errors="coerce").fillna(0)

    # Final sort (descending)
    grouped = grouped.sort_values(ascending=False)

    # Build labels with name + value + percent
    total = float(grouped.sum())
    labels = grouped.index.astype(str).tolist()
    sizes = grouped.values.tolist()
    pct = [(s / total) * 100.0 if total else 0.0 for s in sizes]

    # Build readable label text: "Name\nvalue (rounded)\nxx.x%"

    def fmt_val(v):
        # integer-like -> no decimals
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}"

    # --- Clean labels: show ONLY percent on wedges (no names/values) ---
    pct_labels = [f"{p:.1f}%" for p in pct]

    fig = plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        sizes,
        radius= 1.3,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        pctdistance=0.78,
        textprops={"fontsize": 9},
        wedgeprops=dict(width=0.60, edgecolor="white")
    )
    # --- Center Label (shows total value) ---
    total_sum = sum(sizes)

    def fmt_money_center(v):
        if abs(v) >= 1e9:
            val = v / 1e9
            return f"${val:.2f}B" 
        if abs(v) >= 1e6:
            val = v / 1e6
            return f"${val:.2f}M"
        if abs(v) >= 1e3:
            val = v / 1e3
            return f"${val:.2f}k"
        return f"${v:,.2f}"


    center_label = f"Total:\n{fmt_money_center(total_sum)}"

    plt.text(
        0, 0,                     # absolute center of pie
        center_label,
        ha='center',
        va='center',
        fontsize=12,
    )


    # Move percentage labels outward slightly
    for t in autotexts:
        t.set_color("black")

    # Legend for category names
    plt.legend(
        labels,
        title=x_col,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
    )

    # Add a value table on the right (no overlap)
    value_lines = [
        f"{lbl}: {fmt_val(val)}"
        for lbl, val in zip(labels, sizes)
    ]
    
    plt.text(
        1.25, 0.1,
        "\n".join(value_lines),
        transform=plt.gca().transAxes,
        fontsize=9,
        va="bottom"
    )

    plt.title(f"Share of {y_col} by {x_col}", fontsize=12, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    return plt.gcf()




def _beautify_axes(ax, is_time_x=False):
    """Apply consistent modern styling to an Axes object."""
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#ffffff")
    ax.tick_params(axis="both", labelsize=10)

    # ✅ Only apply default numeric formatter if no custom formatter was set
    current_formatter = ax.yaxis.get_major_formatter()
    if isinstance(current_formatter, mtick.ScalarFormatter):
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(
                lambda x, pos: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"
            )
        )

    # Optionally format x as dates if requested
    if is_time_x:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")


def _limit_xticks(ax, max_labels=12):
    """Reduce number of xtick labels if too many categories."""
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()
    n = len(labels)
    if n > max_labels:
        step = max(1, int(n / max_labels))
        for i, lbl in enumerate(labels):
            if i % step != 0:
                lbl.set_visible(False)
            else:
                lbl.set_visible(True)
        # tighten layout will handle spacing


# ---------------- Chart Creation ---------------- #

def create_chart_image(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str = "bar"):
    import matplotlib
    matplotlib.use("Agg")
    # removed local imports of pandas/numpy/plt to avoid UnboundLocalError
    # rely on module-level imports at top of file

    max_plot_rows = 2000
    
    if chart_type == "heatmap":
        # ---------- 1. Select numeric columns ----------
        num_df = df.select_dtypes(include=[np.number]).copy()
        candidate_cols = list(num_df.columns)
        clean_cols = []
        n_rows = len(df)

        for c in candidate_cols:
            s = df[c]

            # A) Exclude near-unique columns (ID-like)
            nunq = s.nunique(dropna=True)
            if n_rows > 0 and (nunq / max(1, n_rows) > 0.9):
                continue

            # B) Exclude date-like numeric columns (Excel serials / timestamps)
            try:
                dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                if dt.notna().sum() >= max(5, int(0.5 * len(s))):
                    continue
            except Exception:
                pass

            # C) Exclude monotonic integer columns (counters, auto IDs)
            if pd.api.types.is_integer_dtype(s):
                diffs = s.diff().dropna()
                if len(diffs) > 5 and (diffs >= 0).mean() > 0.95 and nunq > 10:
                    continue

            # Passed all checks → keep as real metric
            clean_cols.append(c)

        # ---------- 2. Fallback if too strict ----------
        if len(clean_cols) < 2:
            clean_cols = [
                c for c in candidate_cols
                if not any(k in c.lower() for k in ("date", "time", "timestamp", "id"))
            ]

        if len(clean_cols) < 2:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Need at least 2 usable numeric columns for heatmap", ha="center")
            plt.tight_layout()
            return _to_img_base64()

        sel = df[clean_cols].copy()

        # ---------- 3. Correlation matrix only (ALWAYS) ----------
        matrix = sel.corr(numeric_only=True)

        # ---------- 4. Plot ----------
        plt.figure(
            figsize=(max(8, 0.8 * len(clean_cols)), max(6, 0.8 * len(clean_cols))),
            dpi=120
        )
        im = plt.imshow(matrix.values, cmap="coolwarm", interpolation="nearest", aspect="auto")
        ax = plt.gca()

        # Ticks / labels
        ax.set_xticks(range(len(clean_cols)))
        ax.set_yticks(range(len(clean_cols)))
        ax.set_xticklabels(clean_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(clean_cols, fontsize=9)

        # Colorbar (simple float formatting)
        cbar = plt.colorbar(im, fraction=0.04, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, p: f"{v:.2f}"))

        # Cell annotations
        for (i, j), val in np.ndenumerate(matrix.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _to_img_base64()


    # --- Scatter Matrix ---
    if chart_type == "scatter_matrix":
        num_df = df.select_dtypes(include=[np.number])

        # Remove date-like numeric columns
        date_like_cols = [c for c in num_df.columns if "date" in c.lower() or "id" in c.lower()]
        num_df = num_df.drop(columns=date_like_cols, errors="ignore")

        if num_df.shape[1] < 2:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, "Need at least 2 numeric columns for scatter matrix", ha="center")
            plt.tight_layout()
            return _to_img_base64()

        # sample large datasets
        if len(num_df) > 1000:
            num_df = num_df.sample(1000, random_state=42)

        # Bigger figure for cleaner view
        axs = scatter_matrix(num_df, figsize=(18, 14), diagonal='hist',
                            color='#4e79a7', alpha=0.6)
        
        # --- Clean up diagonal hist bars (fix merged-looking bars) ---
        for idx in range(len(num_df.columns)):
            ax = axs[idx, idx]      # diagonal histogram axis
            for patch in ax.patches:
                patch.set_edgecolor("black")   # separate each bar
                patch.set_linewidth(0.7)


        # Money formatter
        money_keys = ("profit", "revenue", "cost", "sales", "price")
        money_cols = [c for c in num_df.columns if any(k in c.lower() for k in money_keys)]

        def money_formatter(x, pos):
            if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
            if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
            return f"${x:,.0f}"

        cols = list(num_df.columns)
        n = len(cols)
        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                if cols[j] in money_cols:
                    ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
                if cols[i] in money_cols:
                    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(30)
                    lbl.set_ha("right")

        plt.suptitle("Scatter Matrix of Numeric Features", fontsize=18, fontweight="bold")
        plt.tight_layout()
        return _to_img_base64()


    # proceed as before when x_col/y_col are provided
    df_plot = None
    if x_col is not None and y_col is not None:
        df_plot = df[[x_col, y_col]].copy()
    elif x_col is None and y_col is not None:
        df_plot = df[[y_col]].copy()
    elif x_col is not None and y_col is None:
        df_plot = df[[x_col]].copy()
    else:
        df_plot = df.copy()
    # If both names identical, keep one copy (existing behavior)
    if x_col == y_col and x_col is not None:
        df_plot = df[[y_col]].copy()


    if chart_type in ("bar", "line", "scatter", "box", "hist", "time"):
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")

    plt.figure(figsize=(6, 4), dpi=80)

    def looks_like_time(series):
        try:
            pd.to_datetime(series.dropna().iloc[:10], errors="raise")
            return True
        except Exception:
            return False

    x_is_time = looks_like_time(df_plot[x_col])



    if chart_type == "box":
        try:
            # -----------------------------
            #  BASIC PREP
            # -----------------------------
            if x_col == y_col:
                df_plot = df[[y_col]].copy()
            else:
                df_plot = df[[x_col, y_col]].copy()
                # 🔄 Auto-detect roles:
                if pd.to_numeric(df_plot[x_col], errors="coerce").notna().sum() > \
                pd.to_numeric(df_plot[y_col], errors="coerce").notna().sum():
                    x_col, y_col = y_col, x_col  # Swap so Y is numeric and X is category

            df_plot = df_plot.dropna(subset=[y_col])
            if df_plot.empty:
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, "No data available.", ha="center")
                return _to_img_base64()

            # y must be numeric
            y_vals = pd.to_numeric(df_plot[y_col], errors="coerce")
            df_plot[y_col] = y_vals
            if y_vals.notna().sum() < 2:
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, f"No numeric values found in {y_col}.", ha="center")
                return _to_img_base64()

            # sampling for performance
            MAX_ROWS = 5000
            if len(df_plot) > MAX_ROWS:
                df_plot = df_plot.sample(MAX_ROWS, random_state=42)

            fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

            # Money formatter
            def money_formatter(x, pos):
                if abs(x) >= 1e9:
                    return f"${x/1e9:,.1f}B"
                if abs(x) >= 1e6:
                    return f"${x/1e6:,.1f}M"
                if abs(x) >= 1e3:
                    return f"${x/1e3:,.0f}k"
                return f"${x:,.0f}"

            # -----------------------------
            # CASE 1: ONLY Y PROVIDED → single box
            # (also covers x_col == y_col from caller)
            # -----------------------------
            if (not x_col) or (x_col == y_col):
                # force clean 1-D numeric array
                data = pd.to_numeric(df_plot[y_col], errors="coerce").dropna().values
                if data.ndim != 1:
                    data = data.ravel()

                # matplotlib expects a *sequence of datasets* → wrap in a list
                bp = ax.boxplot([data], labels=[y_col], patch_artist=True)
                print("BOX CASE1 len(data) =", len(data), "dtype=", data.dtype)
                print("First few values:", data[:5])


                for patch in bp['boxes']:
                    patch.set_facecolor("#4e79a7")

                ax.set_title(f"Box Plot of {y_col}", fontsize=14)
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
                _beautify_axes(ax, is_time_x=False)
                return _to_img_base64()




            # -----------------------------
            # CASE 2: BOTH NUMERIC COLUMNS → compare distributions
            # -----------------------------
            x_num = pd.to_numeric(df_plot[x_col], errors="coerce")
            x_is_numeric = x_num.notna().sum() >= 0.1 * len(df_plot)
            y_is_numeric = True  # already checked

            if x_is_numeric and y_is_numeric:
                # Two numeric columns → show 2 boxes
                bx_data = [df_plot[y_col].dropna().values, x_num.dropna().values]
                bx_labels = [y_col, x_col]
                bp = ax.boxplot(bx_data, labels=bx_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor("#6a3d9a")
                ax.set_title(f"Box Plot of {y_col} and {x_col}", fontsize=14)
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
                _beautify_axes(ax, is_time_x=False)
                return _to_img_base64()

            # -----------------------------
            # CASE 3: X CATEGORICAL, Y NUMERIC
            # -----------------------------
            df_plot[x_col] = df_plot[x_col].astype(str)
            grouped = df_plot.groupby(x_col)[y_col].apply(lambda s: s.dropna().values)

            # remove empty groups
            grouped = grouped[grouped.apply(lambda arr: len(arr) > 0)]
            if grouped.empty:
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, "No valid groups to plot.", ha="center")
                return _to_img_base64()

            # keep top N categories
            TOP_N = 12
            sums = grouped.apply(lambda a: a.sum())
            top_keys = sums.sort_values(ascending=False).index[:TOP_N]
            final_groups = grouped.loc[top_keys]

            # combine rest to "Other"
            if len(grouped) > TOP_N:
                others = grouped.drop(top_keys)
                other_vals = np.concatenate([v for v in others.values])
                if len(other_vals) > 0:
                    final_groups = pd.concat([final_groups, pd.Series({"Other": other_vals})])

            labels = list(final_groups.index)
            data = list(final_groups.values)

            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor("#4e79a7")

            plt.xticks(rotation=35, ha="right")
            ax.set_title(f"Box Plot of {y_col} by {x_col}", fontsize=15)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
            _limit_xticks(ax, max_labels=20)
            _beautify_axes(ax, is_time_x=False)
            return _to_img_base64()

        except Exception as e:
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha="center")
            return _to_img_base64()



    elif chart_type == "scatter":
        # --- Auto-swap for scatter if needed ---
        if not pd.api.types.is_numeric_dtype(df_plot[x_col]) and pd.api.types.is_numeric_dtype(df_plot[y_col]):
            x_col, y_col = y_col, x_col

        # --- Skip non-numeric scatter safely ---
        if not (pd.api.types.is_numeric_dtype(df_plot[x_col]) and pd.api.types.is_numeric_dtype(df_plot[y_col])):
            plt.text(0.5, 0.5, "Scatter plot requires numeric columns", ha="center")
        else:
            plt.scatter(df_plot[x_col], df_plot[y_col], s=10)
            # Money formatter for Y-axis
            def money_formatter(x, pos):
                if abs(x) >= 1e9:
                    return f"${x/1e9:.1f}B"
                if abs(x) >= 1e6:
                    return f"${x/1e6:.1f}M"
                if abs(x) >= 1e3:
                    return f"${x/1e3:.0f}k"
                return f"${x:,.0f}"

            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

            # X-axis money formatting too (if numeric)
            plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Scatter Plot of {x_col} vs {y_col}")

    elif chart_type == "line":
        try:
            # Basic clean
            df_plot = df_plot.dropna(subset=[x_col, y_col]).copy()
            
            # ---------- NEW: auto-detect which column is the date ----------
            x_dt_candidate = _series_to_datetime_flex(df_plot[x_col])
            y_dt_candidate = _series_to_datetime_flex(df_plot[y_col])

            n = len(df_plot)
            x_is_date = x_dt_candidate.notna().sum() >= 0.5 * n
            y_is_date = y_dt_candidate.notna().sum() >= 0.5 * n

            # If Y looks like date but X doesn't → swap roles
            if (not x_is_date) and y_is_date:
                x_col, y_col = y_col, x_col
                x_dt_candidate = y_dt_candidate
            # (if neither or both look like dates, keep original order)

            # Robust datetime conversion
            df_plot['_x_dt'] = _series_to_datetime_flex(df_plot[x_col])
            df_plot = df_plot.dropna(subset=['_x_dt']).copy()
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")
            df_plot = df_plot.dropna(subset=[y_col]).copy()

            # Sort time
            df_plot = df_plot.sort_values('_x_dt')

            # Smart aggregation level
            span_days = (df_plot['_x_dt'].max() - df_plot['_x_dt'].min()).days
            if span_days > 365 * 3:
                df_plot['_period'] = df_plot['_x_dt'].dt.to_period('Y').dt.to_timestamp()
                title_suffix = " (Yearly Total)"
            elif span_days > 180:
                df_plot['_period'] = df_plot['_x_dt'].dt.to_period('M').dt.to_timestamp()
                title_suffix = " (Monthly Total)"
            else:
                df_plot['_period'] = df_plot['_x_dt'].dt.floor('D')
                title_suffix = " (Daily Total)"

            grouped = df_plot.groupby('_period', dropna=False)[y_col].sum()

            # Downsampling only when needed
            if len(grouped) > 500:
                step = max(1, len(grouped)//500)
                grouped = grouped.iloc[::step]

            # Plot
            fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
            ax.plot(grouped.index, grouped.values, linewidth=1.8)

            # Title & labels
            ax.set_title(f"Line Chart of {y_col} by {x_col}{title_suffix}", fontsize=12, fontweight="bold")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

            # Date formatting
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Dynamic numeric formatter → universal
            def smart_format(x, pos):
                if abs(x) >= 1e9: return f"{x/1e9:,.1f}B"
                if abs(x) >= 1e6: return f"{x/1e6:,.1f}M"
                if abs(x) >= 1e3: return f"{x/1e3:,.0f}K"
                return f"{x:,.0f}"
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(smart_format))

            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            _beautify_axes(ax, is_time_x=True)

            return _to_img_base64()

        except Exception as e:
            plt.figure(figsize=(9, 4))
            plt.text(0.5, 0.5, f"Line Chart Error: {e}", ha="center")
            return _to_img_base64()



    elif chart_type == "bar":
        # Robust bar plotting: detect numeric columns by coercion (data-driven)
        try:
            # prepare and clean
            df_plot = df[[x_col, y_col]].copy()
            df_plot = df_plot.dropna(how="all").dropna(subset=[y_col], how="all")
            if df_plot.shape[0] == 0:
                plt.figure(figsize=(10, 4), dpi=100)
                plt.text(0.5, 0.5, "No data to plot (empty after filtering).", ha="center")
                plt.title("Bar chart (no data)")
                return plt.gcf()

            # Try coercion to numeric to decide branches (do not overwrite original columns)
            x_as_num = pd.to_numeric(df_plot[x_col], errors="coerce")
            y_as_num = pd.to_numeric(df_plot[y_col], errors="coerce")
            def money_formatter(x, pos):
                # show in billions if > 1e9
                if abs(x) >= 1e9:
                    return f"${x/1e9:,.1f}B"
                if abs(x) >= 1e6:
                    return f"${x/1e6:,.1f}M"
                if abs(x) >= 1e3:
                    return f"${x/1e3:,.0f}k"
                return f"${x:,.0f}"
            


            # How many valid numeric rows exist
            n = len(df_plot)
            x_num_count = int(x_as_num.notna().sum())
            y_num_count = int(y_as_num.notna().sum())

            # Heuristics: consider a column numeric if at least 10% of rows parse to numeric or at least 1 value
            def is_numeric_like(count):
                return (count >= max(1, int(0.10 * n)))

            x_is_numeric = is_numeric_like(x_num_count)
            y_is_numeric = is_numeric_like(y_num_count)

            # If both numeric-like -> treat as numeric vs numeric comparison (aggregate)
            fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

            # Format Y axis with currency—smart formatter (shows B for billions)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

            if x_is_numeric and y_is_numeric:
                # numeric-numeric: aggregate sums and show two bars (clean comparison)
                sum_x = x_as_num.sum()
                sum_y = y_as_num.sum()
                keys = [x_col, y_col]
                vals = [sum_x, sum_y]
                colors = ["#f28e2b", "#59a14f"]
                ax.bar(keys, vals, color=colors, width=0.6, edgecolor="k", linewidth=0.3)
                ax.set_title(f"Comparison of {x_col} and {y_col}", fontsize=16, fontweight="bold", pad=14)
                ax.set_xlabel("Metric", fontsize=12)
                ax.set_ylabel(f"Amount (in billions)", fontsize=12)
                _beautify_axes(ax, is_time_x=False)

            elif (not x_is_numeric) and y_is_numeric:
                # categorical x, numeric y: group & show top categories + Other
                grouped = df_plot.groupby(df_plot[x_col].astype(str), sort=False)[y_col].sum()
                if len(grouped) == 0:
                    plt.text(0.5, 0.5, "No groups to plot.", ha="center")
                else:
                    # keep top_n categories
                    top_n = 30
                    grouped_sorted = grouped.sort_values(ascending=False)
                    if len(grouped_sorted) > top_n:
                        top = grouped_sorted.iloc[:top_n]
                        other_sum = grouped_sorted.iloc[top_n:].sum()
                        grouped_final = pd.concat([top, pd.Series({"Other": other_sum})])
                    else:
                        grouped_final = grouped_sorted
                    grouped_final = grouped_final.sort_values(ascending=False)

                    grouped_final.plot(kind="bar", ax=ax, color="#4e79a7", width=0.75, edgecolor="k", linewidth=0.2)
                    ax.set_title(f"Bar Chart of {y_col} by {x_col}", fontsize=16, fontweight="bold", pad=14)
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    plt.xticks(rotation=35, ha="right")
                    _limit_xticks(ax, max_labels=16)
                    _beautify_axes(ax, is_time_x=False)

            elif x_is_numeric and (not y_is_numeric):
                # uncommon: x numeric, y non-numeric -> aggregate counts per x
                counts = df_plot.groupby(df_plot[x_col])[y_col].count().sort_values(ascending=False)
                counts.plot(kind="bar", ax=ax, color="#4e79a7", width=0.75)
                ax.set_title(f"Counts of {y_col} per {x_col}", fontsize=16, fontweight="bold", pad=14)
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                _limit_xticks(ax, max_labels=20)
                _beautify_axes(ax, is_time_x=False)

            else:
                # fallback: try plotting sample row-level bars but avoid huge plot
                max_bars = 1000
                if len(df_plot) > max_bars:
                    sample_df = df_plot.sample(n=max_bars, random_state=42)
                else:
                    sample_df = df_plot
                # try numeric coercion for y just in case
                y_vals = pd.to_numeric(sample_df[y_col], errors="coerce")
                ax.bar(sample_df[x_col].astype(str), y_vals, color="#4e79a7", width=0.6)
                ax.set_title(f"Bar Chart of {y_col} vs {x_col}", fontsize=16, fontweight="bold", pad=14)
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                plt.xticks(rotation=35, ha="right")
                _limit_xticks(ax, max_labels=20)
                _beautify_axes(ax, is_time_x=False)

            plt.tight_layout()
            plt.sca(ax)
        except Exception as e:
            plt.figure(figsize=(10, 4), dpi=100)
            plt.text(0.5, 0.5, f"⚠️ Unable to create bar chart: {str(e)}", ha="center")
            plt.title("Bar chart error")

    
    elif chart_type == "pie":
        fig = create_pie_chart(df, x_col, y_col)
        plt.figure(fig.number)
    


    elif chart_type == "hist":
        # --- choose numeric column safely ---
        numeric_candidates = []
        for c in [y_col, x_col]:
            if c in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[c]):
                numeric_candidates.append(c)

        if not numeric_candidates:
            plt.text(0.5, 0.5, "Histogram requires numeric column", ha="center")
            return _to_img_base64()

        col = numeric_candidates[0]
        data = pd.to_numeric(df_plot[col], errors="coerce").dropna()
        if len(data) < 10:
            plt.text(0.5, 0.5, "Not enough numeric data to plot.", ha="center")
            return _to_img_base64()

        # --- robust binning + outlier handling ---
        # clamp minimum at 0 for money-like fields
        min_val = max(0, float(data.min()))
        max_val = float(data.max())

        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1 if q3 >= q1 else 0.0

        # clip extreme right tail if very skewed
        if iqr > 0 and max_val > q3 + 3 * iqr:
            upper = np.percentile(data, 99)  # keep 99% of points
            data = data[data <= upper]
            max_val = float(data.max())

        # Freedman–Diaconis bin rule (bounded between 20 and 60)
        if iqr > 0:
            bin_width = 2 * iqr / (len(data) ** (1 / 3))
            if bin_width > 0:
                est_bins = int((max_val - min_val) / bin_width)
            else:
                est_bins = 30
        else:
            est_bins = 30

        bins = max(20, min(60, est_bins))

        # --- draw nice, clean histogram on a fresh big figure ---
        plt.clf()
        fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
        ax.hist(data, bins=bins, range=(min_val, max_val),
                color="#2E7D32", edgecolor="black", alpha=0.85)

        ax.set_title(f"Distribution of {col}", fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=13)
        ax.set_ylabel("Frequency (Number of Orders)", fontsize=13)

        # money-style x-axis formatter
        def money_formatter(x, pos):
            if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
            if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
            return f"${x:,.0f}"

        ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

        ax.grid(axis="y", linestyle="--", alpha=0.35)
        _beautify_axes(ax, is_time_x=False)
        # (no return here; create_chart_image's final plt.tight_layout() + _to_img_base64() will run)

    else:
        plt.text(0.5, 0.5, f"Unsupported chart type: {chart_type}", ha="center")


    plt.tight_layout()
    return _to_img_base64()


# ---------------- Regression ---------------- #

def perform_regression(df: pd.DataFrame, x_col: str, y_col: str):
    X = df[[x_col]].dropna().astype(float).values.reshape(-1, 1)
    y = df[y_col].dropna().astype(float).values
    min_len = min(len(X), len(y))
    if min_len == 0:
        return {"ok": False, "note": "❌ Not enough numeric data for regression."}
    X, y = X[:min_len], y[:min_len]
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    import seaborn as sns

    plt.figure(figsize=(10, 6), dpi=120)

    sns.scatterplot(
        x=X.flatten(),
        y=y,
        s=14,                      
        alpha=0.45,                
        color="#1f77b4",
        edgecolor="white",         
        linewidth=0.3,             
        label="Actual"
    )


    plt.plot(X, model.predict(X), color="red", linewidth=2, label="Predicted")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Regression: {y_col} vs {x_col}")
    plt.legend()

    # ✅ Money formatter (ADD THIS)
    def money_formatter(x, pos):
        if abs(x) >= 1e9:
            return f"${x/1e9:.1f}B"
        if abs(x) >= 1e6:
            return f"${x/1e6:.1f}M"
        if abs(x) >= 1e3:
            return f"${x/1e3:.0f}k"
        return f"${x:,.0f}"

    ax = plt.gca()

    # Y-axis is always dependent variable → format it
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

    # X-axis only if monetary
    if any(k in x_col.lower() for k in ["cost", "price", "revenue", "profit"]):
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

    chart_b64 = _to_img_base64()


    return {
        "ok": True,
        "note": f"📈 Regression: {y_col} = {slope:.4f}*{x_col} + {intercept:.4f}, R²={r2:.3f}",
        "chart": chart_b64,
    }
    
    
# (rest of file unchanged — polynomial/multiple/log regressions and handler)
# ——————————————————————————————————————————————————————————————

    
def perform_polynomial_regression(df: pd.DataFrame, x_col: str, y_col: str, degree: int = 2):
    # numeric cleanup
    X = pd.to_numeric(df[x_col], errors="coerce").values.reshape(-1, 1)
    y = pd.to_numeric(df[y_col], errors="coerce").values

    mask = (~np.isnan(X.flatten())) & (~np.isnan(y))
    X, y = X[mask], y[mask]

    if len(X) < 5:
        return {"ok": False, "note": "❌ Not enough numeric data for polynomial regression."}

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    r2 = model.score(X_poly, y)

    # ---- Plot ----
    plt.figure(figsize=(9, 5), dpi=120)
    plt.scatter(X, y, s=14, alpha=0.5, label="Actual")

    x_sorted = np.sort(X, axis=0)
    y_pred = model.predict(poly.transform(x_sorted))
    plt.plot(x_sorted, y_pred, color="red", linewidth=2, label=f"Polynomial (deg={degree})")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Polynomial Regression (degree {degree})")

    ax = plt.gca()

    # money formatter (Y always)
    def money_formatter(x, pos):
        if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
        if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
        if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
        return f"${x:,.0f}"

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

    plt.legend()
    chart = _to_img_base64()

    return {
        "ok": True,
        "note": f"📈 Polynomial Regression (deg={degree}), R² = {r2:.3f}",
        "chart": chart
    }


def perform_multiple_regression(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    # --- Numeric cleanup ---
    y = pd.to_numeric(df[y_col], errors="coerce")

    X = df[x_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaNs
    combined = pd.concat([y, X], axis=1).dropna()
    y = combined[y_col].values
    X = combined[x_cols].values

    if len(X) < 5:
        return {"ok": False, "note": "❌ Not enough data for multiple regression."}

    # --- Model ---
    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    # --- Build readable equation ---
    terms = [f"{coef:.3f}·{col}" for coef, col in zip(model.coef_, x_cols)]
    equation = f"{y_col} = " + " + ".join(terms) + f" + {model.intercept_:.2f}"

    # --- Plot: Predicted vs Actual ---
    plt.figure(figsize=(9, 5), dpi=120)
    y_pred = model.predict(X)
    plt.scatter(y, y_pred, alpha=0.35, s=10)
    low = min(y.min(), y_pred.min())
    high = max(y.max(), y_pred.max())
    plt.plot([low, high], [low, high], "r--", linewidth=1.5)



    plt.xlabel("Actual " + y_col)
    plt.ylabel("Predicted " + y_col)
    plt.title("Multiple Regression: Actual vs Predicted")

    ax = plt.gca()

    # Money formatter (both axes)
    def money_formatter(x, pos):
        if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
        if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
        if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
        return f"${x:,.0f}"

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

    chart = _to_img_base64()

    return {
        "ok": True,
        "note": f"📊 Multiple Regression\n{equation}\nR² = {r2:.3f}",
        "chart": chart
    }


def perform_log_regression(df: pd.DataFrame, x_col: str, y_col: str):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # --- 1) Prepare & align numeric data ---
    df2 = df[[x_col, y_col]].copy()
    df2[x_col] = pd.to_numeric(df2[x_col], errors="coerce")
    df2[y_col] = pd.to_numeric(df2[y_col], errors="coerce")
    df2 = df2.dropna().reset_index(drop=True)

    if len(df2) < 10:
        return {"ok": False, "note": "❌ Not enough paired numeric rows for log regression (need >=10)."}

    X_raw = df2[x_col].values  # numpy 1-D
    Y_raw = df2[y_col].values  # numpy 1-D

    results = []

    def fit_and_score(X_vals, Y_vals, model_name, eqn_fmt, xlabel, ylabel):
        # Ensure numpy 1-D arrays
        X_vals = np.asarray(X_vals).astype(float).flatten()
        Y_vals = np.asarray(Y_vals).astype(float).flatten()
        if len(X_vals) < 5:
            return None
        model = LinearRegression()
        model.fit(X_vals.reshape(-1, 1), Y_vals)
        r2 = model.score(X_vals.reshape(-1, 1), Y_vals)
        return {
            "name": model_name,
            "model": model,
            "r2": r2,
            "X": X_vals,
            "Y": Y_vals,
            "eqn_fmt": eqn_fmt,
            "xlabel": xlabel,
            "ylabel": ylabel
        }

    # 2) Candidate models (only when valid)
    # Log-Log: ln(Y) ~ ln(X)  (requires X>0 & Y>0)
    if np.all(X_raw > 0) and np.all(Y_raw > 0):
        res = fit_and_score(
            np.log(X_raw),
            np.log(Y_raw),
            "Log–Log",
            "ln(Y) = {b0:.4f} + {b1:.4f} ln(X)",
            f"ln({x_col})",
            f"ln({y_col})"
        )
        if res: results.append(res)

    # Level-Log: Y ~ ln(X)  (requires X>0)
    if np.all(X_raw > 0):
        res = fit_and_score(
            np.log(X_raw),
            Y_raw,
            "Level–Log",
            "Y = {b0:.4f} + {b1:.4f} ln(X)",
            f"ln({x_col})",
            f"{y_col}"
        )
        if res: results.append(res)

    # Log-Level: ln(Y) ~ X  (requires Y>0)
    if np.all(Y_raw > 0):
        res = fit_and_score(
            X_raw,
            np.log(Y_raw),
            "Log–Level",
            "ln(Y) = {b0:.4f} + {b1:.4f} X",
            f"{x_col}",
            f"ln({y_col})"
        )
        if res: results.append(res)

    if not results:
        return {"ok": False, "note": "❌ Log regression not possible: need positive values for X or Y depending on model."}

    # 3) Choose best model by R²
    best = max(results, key=lambda r: r["r2"])
    model = best["model"]
    b0 = float(model.intercept_)
    b1 = float(model.coef_[0])
    r2 = float(best["r2"])

    # 4) Plot (use numpy arrays; ensure predict gets 2D)
    plt.figure(figsize=(10, 6), dpi=120)
    ax = plt.gca()

    X_plot = best["X"]
    Y_plot = best["Y"]

    ax.scatter(X_plot, Y_plot, s=18, alpha=0.5, label="Transformed data", edgecolors="w", linewidth=0.2)

    x_sorted = np.sort(X_plot)
    y_pred = model.predict(x_sorted.reshape(-1, 1))
    ax.plot(x_sorted, y_pred, color="red", linewidth=2, label="Fit")

    # Axis labels & title reflecting transformation
    ax.set_xlabel(best.get("xlabel", "X"))
    ax.set_ylabel(best.get("ylabel", "Y"))
    ax.set_title(f"{best['name']} Regression: {best['ylabel']} vs {best['xlabel']}")

    # Format Y-axis if underlying original Y looks monetary
    if any(k in y_col.lower() for k in ["profit", "revenue", "cost", "sales", "price"]):
        def money_formatter(x, pos):
            if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
            if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
            return f"${x:,.0f}"
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_formatter))

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    chart = _to_img_base64()

    # 5) Nicely formatted equation & interpretation
    # Display equation using column names only (no X/Y, no interpretation text)
    if best["name"] == "Log–Log":
        eqn_text = f"ln({y_col}) = {b0:.4f} + {b1:.4f} · ln({x_col})"
        note = (
            f"📈 {best['name']} selected | R² = {r2:.3f}\n"
            f"Equation: {eqn_text}\n"
            f"Elasticity: {b1:.3f}"
        )

    elif best["name"] == "Level–Log":
        eqn_text = f"{y_col} = {b0:.4f} + {b1:.4f} · ln({x_col})"
        note = f"📈 {best['name']} selected | R² = {r2:.3f}\nEquation: {eqn_text}"

    elif best["name"] == "Log–Level":
        eqn_text = f"ln({y_col}) = {b0:.4f} + {b1:.4f} · {x_col}"
        note = f"📈 {best['name']} selected | R² = {r2:.3f}\nEquation: {eqn_text}"


    return {"ok": True, "note": note, "chart": chart}




# ---------------- Main Handler ---------------- #

def handle_excel_task(intent: str, df: pd.DataFrame, message: str):
    """
    Main AI Excel function router.
    Handles supported intents + conditional + descriptive logic.
    """
        # --- Clean the dataframe to ignore previously placed AI result rows (numeric or summary) ---
    df = df.copy()

    # Drop completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    try:
        from nlp_utils import extract_condition_from_message
    except Exception:
        extract_condition_from_message = lambda m, d: {}

    try:
        import re
        if "multiple" in message.lower() and " on " in message.lower():
            cols = extract_columns_from_message(message, df, max_cols=None)
        else:
            cols = extract_columns_from_message(message, df)

        print(f"🧠 Extracted columns (initial): {cols}")


        # If user didn’t specify {columns}, allow chart types that don't require braces
        if not cols:
            # detect intent-based chart that can run without explicit brace columns
            chart_type_guess = detect_chart_type(message)
            allowed_no_cols = {"scatter_matrix", "heatmap"}
            if chart_type_guess in allowed_no_cols:
                # proceed (cols stay empty) so chart branch can handle defaults / all-numeric
                pass
            else:
                return {"ok": False, "note": "⚠️ Please specify column(s) in braces, e.g. sum {Total Cost} or correlation {Sales} and {Profit}."}


        # Primary/secondary columns (for two-column charts/correlation)
        col = cols[0] if cols else None
        col2 = cols[1] if len(cols) > 1 else None

        # show a sample of the raw column values and types
            ##
        condition = {}
        try:
            condition = extract_condition_from_message(message, df)
        except Exception:
            condition = {}

        df_filtered = df.copy()

        # Apply filters (support numeric and string comparisons)
        # Apply filters (support numeric, date and string comparisons)
        if condition:
            # normalize condition keys to actual df column names (case-insensitive, fuzzy fallback)
            normalized_condition = {}
            df_col_lower_map = {str(c).lower(): c for c in df.columns}

            for raw_col, (op, val) in condition.items():
                if raw_col is None:
                    continue
                raw_col_s = str(raw_col).strip()
                # exact (case-insensitive)
                match_col = df_col_lower_map.get(raw_col_s.lower())
                # fuzzy fallback
                if match_col is None:
                    close = difflib.get_close_matches(raw_col_s.lower(), list(df_col_lower_map.keys()), n=1, cutoff=0.7)
                    if close:
                        match_col = df_col_lower_map[close[0]]
                if match_col:
                    normalized_condition[match_col] = (op, val)

            # Ensure df_filtered exists BEFORE the loop
            df_filtered = df.copy()

            # now apply normalized_condition
            for col_name, (op, val) in normalized_condition.items():
                if col_name not in df_filtered.columns:
                    continue

                # normalize operator (is/equals/== → =)
                op = (op or "").strip().lower()
                if op in ("is", "equals", "equal to", "=="):
                    op = "="

                series_orig = df_filtered[col_name]
                val_str = str(val).strip().strip("'\"")
                applied = False

                # ---- 1) DATE FIRST (robust) ----
                val_dt = _parse_date_flex(val_str)
                if pd.notna(val_dt):
                    series_dt = _series_to_datetime_flex(series_orig)
                    # DEBUG: show what we’re comparing
                    try:
                        sample_vals = series_dt.dropna().unique()[:3]
                    except Exception:
                        sample_vals = []
                    print(f"[COND:DATE] col='{col_name}' op='{op}' val='{val_str}' parsed={val_dt.date()} dtype={series_orig.dtype} sample={sample_vals}")

                    if op == "=":
                        mask = (series_dt == val_dt)
                    elif op == ">":
                        mask = (series_dt > val_dt)
                    elif op == "<":
                        mask = (series_dt < val_dt)
                    elif op == ">=":
                        mask = (series_dt >= val_dt)
                    elif op == "<=":
                        mask = (series_dt <= val_dt)
                    else:
                        mask = (series_dt == val_dt)  # default equality

                    df_filtered = df_filtered[mask]
                    applied = True

                if applied:
                    continue

                # ---- 2) NUMERIC ----
                try:
                    val_num = float(val_str)
                    series_num = pd.to_numeric(series_orig, errors="coerce")
                    print(f"[COND:NUM] col='{col_name}' op='{op}' val='{val_str}' parsed={val_num} dtype={series_orig.dtype}")
                    if op == "=":
                        mask = (series_num == val_num)
                    elif op == ">":
                        mask = (series_num > val_num)
                    elif op == "<":
                        mask = (series_num < val_num)
                    elif op == ">=":
                        mask = (series_num >= val_num)
                    elif op == "<=":
                        mask = (series_num <= val_num)
                    else:
                        mask = (series_num == val_num)

                    df_filtered = df_filtered[mask]
                    applied = True
                except Exception:
                    applied = False

                if applied:
                    continue

                # ---- 3) STRING (case-insensitive) ----
                series_str = series_orig.astype(str).str.strip().str.lower()
                print(f"[COND:STR] col='{col_name}' op='{op}' val='{val_str.lower()}'")
                if op == "=":
                    mask = (series_str == val_str.lower())
                else:
                    mask = series_str.str.contains(val_str.lower(), na=False)

                df_filtered = df_filtered[mask]

            print(f"🧪 Rows after filtering: {len(df_filtered)}")

            # validate selected column exists after filters
            if col not in df_filtered.columns:
                return {"ok": False, "note": f"⚠️ Column '{col}' not found after filtering."}



        # Basic operations (intent values must match those set by main.py)
        intent_l = (intent or "").lower()
        if intent_l in ["sum", "total", "average", "mean", "median", "mode",
                  "variance", "var", "std", "stddev", "standard deviation",
                  "min", "minimum", "max", "maximum", "range", "difference"]:
            if cols:
                col = cols[0]
                if col not in df_filtered.columns:
                    return {"ok": False, "note": f"⚠️ Column {col} not found after filtering."}
                note = safe_numeric_stat(df_filtered, col, intent_l)
                return {"ok": True, "note": note}

            else:
                return {"ok": False, "note": "⚠️ Could not detect a numeric column."}

            
        elif intent_l in ["count", "rows"]:
            val = len(df_filtered)
            return {"ok": True, "note": f"📄 Total rows = {val}"}
            
        elif intent_l in ["correlation", "relation", "relationship", "compare"]:
            # 1) Prefer brace-based columns if user provided them
            if len(cols) >= 2:
                col1, col2 = cols[:2]
                if col1 not in df_filtered.columns or col2 not in df_filtered.columns:
                    return {"ok": False, "note": "⚠️ Columns not found for correlation after filtering."}
            else:
                # 2) Your original fallback when braces not given
                message_lower = message.lower()
                colnames = []
                for c in df.columns:
                    if c.lower() in message_lower:
                        colnames.append(c)
                if len(colnames) < 2:
                    import re
                    m = re.search(r"(?:between|of|vs)\s+([\w\s]+?)\s+(?:and|&|vs)\s+([\w\s]+)", message_lower)
                    if m:
                        colnames = [m.group(1).strip().title(), m.group(2).strip().title()]
                valid_cols = [c for c in colnames if c in df.columns]
                if len(valid_cols) < 2:
                    return {"ok": False, "note": "⚠️ Couldn't identify two valid numerical columns for correlation."}
                col1, col2 = valid_cols[:2]

            try:
                s1 = pd.to_numeric(df_filtered[col1], errors="coerce")
                s2 = pd.to_numeric(df_filtered[col2], errors="coerce")
                corr = s1.corr(s2)
                if pd.isna(corr):
                    return {"ok": False, "note": f"⚠️ Correlation could not be computed between {col1} and {col2}."}
                return {"ok": True, "note": f"📈 Correlation between {col1} and {col2}: {corr:.4f}"}
            except Exception as e:
                return {"ok": False, "note": f"⚠️ Error computing correlation: {str(e)}"}




        elif any(word in message.lower() for word in ["summary", "statistics", "describe", "overview"]):
            if cols:
                col = cols[0]
                desc = df_filtered[col].describe().to_dict()
                note = "📊 Summary statistics for " + col + ":\n"
                for k, v in desc.items():
                    note += f"{k}: {v}\n"
                return {"ok": True, "note": note.strip()}
            else:
                desc = df_filtered.describe().to_dict()
                return {"ok": True, "note": f"📊 Dataset summary:\n{desc}"}

        elif "compare" in message.lower() and len(cols) >= 2:
            c1, c2 = cols[:2]
            diff = df_filtered[c1] - df_filtered[c2]
            note = f"📉 Comparison ({c1} - {c2}):\nMean diff = {diff.mean():.2f}, Min = {diff.min():.2f}, Max = {diff.max():.2f}"
            return {"ok": True, "note": note}

        elif intent_l in ["chart", "create_chart", "plot"]:
            chart_type = detect_chart_type(message)
            
            # --- Heatmap NEVER requires x_col or y_col ---
            if chart_type == "heatmap":
                img_b64 = create_chart_image(df_filtered, None, None, chart_type)
                return {"ok": True, "note": "📊 Heatmap created", "chart": img_b64}

            # ✅ SPECIAL CASE: scatter-matrix never needs x_col or y_col
            if chart_type == "scatter_matrix":
                img_b64 = create_chart_image(df_filtered, cols[0] if cols else None, 
                                            cols[1] if len(cols) > 1 else None, 
                                            chart_type)
                return {"ok": True, "note": "📊 Scatter Matrix created", "chart": img_b64}

            # ✅ Two columns → normal charts (scatter, bar, line, etc.)
            if len(cols) >= 2:
                img_b64 = create_chart_image(df_filtered, cols[0], cols[1], chart_type)
                return {"ok": True, "note": f"📊 {chart_type.capitalize()} created for {cols[0]} vs {cols[1]}", "chart": img_b64}

            # ✅ One column allowed for: box / hist / pie
            if len(cols) == 1 and chart_type in ["box", "hist", "pie"]:
                img_b64 = create_chart_image(df_filtered, cols[0], cols[0], chart_type)
                return {"ok": True, "note": f"📊 {chart_type.capitalize()} created for {cols[0]}", "chart": img_b64}

            # ❌ Everything else truly needs 2 columns
            return {"ok": False, "note": "❌ Not enough columns to create chart"}



        elif intent_l in ["regression", "predict"]:
            msg = message.lower()

            # MULTIPLE REGRESSION: Y on X1, X2, ...
            if "multiple" in msg and " on " in msg and len(cols) >= 3:
                y_col = cols[0]

                # ✅ keep only numeric predictors
                x_cols = [
                    c for c in cols[1:]
                    if pd.api.types.is_numeric_dtype(df_filtered[c])
                ]

                if len(x_cols) < 1:
                    return {"ok": False, "note": "❌ No numeric predictors available for multiple regression."}

                return perform_multiple_regression(df_filtered, y_col, x_cols)


            # POLYNOMIAL
            if "polynomial" in msg and len(cols) >= 2:
                return perform_polynomial_regression(df_filtered, cols[0], cols[1])

            # LOG
            if "log" in msg and len(cols) >= 2:
                return perform_log_regression(df_filtered, cols[0], cols[1])

            # SIMPLE REGRESSION
            if len(cols) >= 2:
                return perform_regression(df_filtered, cols[0], cols[1])


            else:
                return {"ok": False, "note": "❌ Need two numeric columns for regression"}

        else:
            return {"ok": False, "note": f"🤔 Sorry, I can’t handle '{intent}' yet"}

    except Exception as e:
        return {"ok": False, "note": f"❌ Error while handling: {str(e)}"}
