from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# CONFIG
# =========================================================
BASE_YEAR = 2026
LATEST_YEAR = 2026
HISTORICAL_YEARS = [2024, 2025]
ALL_YEARS = [2024, 2025, 2026]
DATA_ROOT = Path("data")

PROVINCES = [
    "BMC", "BTB", "KP", "KPC", "KPS",
    "KT", "MDK", "PV", "RTK", "SHV",
    "SR", "ST", "SVR", "TBK", "TK"
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

MONTH_MAP = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December",
}
MONTH_FULL_LIST = list(MONTH_MAP.values())

# =========================================================
# PDF TABLE TITLE KHMER FONT CONFIG
# =========================================================
# This Khmer font is used ONLY for the PDF table/title headings.
# Other PDF text and table values remain unchanged.
PDF_TABLE_TITLE_KHMER_FONT_PATH = Path("fonts/KhmerOS_siemreap.ttf")

# Change only these PDF table titles when needed.
PDF_SUMMARY_TABLE_TITLE_TEXT = "តារាងសង្ខេប — 2026, 2025, 2024"
PDF_KPI_TABLE_TITLE_TEXT = "ការប្រៀបធៀប KPI ប្រចាំឆ្នាំ"

st.set_page_config(
    page_title="EDC Cabin Loss Dashboard",
    page_icon="⚡",
    layout="wide",
)

def inject_dashboard_css():
    """Small UI polish for a cleaner and more convenient dashboard."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.0rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 0.75rem 0.85rem;
            border-radius: 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
        }
        .edc-hero {
            padding: 1.0rem 1.2rem;
            border-radius: 1.0rem;
            border: 1px solid #e2e8f0;
            background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
            margin-bottom: 1.0rem;
        }
        .edc-hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2.0rem;
        }
        .edc-hero p {
            margin: 0.35rem 0 0 0;
            color: #475569;
            font-size: 0.98rem;
        }
        .edc-card {
            border: 1px solid #e2e8f0;
            border-radius: 0.9rem;
            padding: 0.85rem 1.0rem;
            background: #ffffff;
            margin-bottom: 0.7rem;
        }
        .edc-small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }
        div.stButton > button {
            border-radius: 0.75rem;
            font-weight: 600;
        }
        div.stDownloadButton > button {
            border-radius: 0.75rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, body: str = ""):
    """Reusable small section intro."""
    body_html = f"<p>{body}</p>" if body else ""
    st.markdown(
        f"""
        <div class="edc-card">
            <strong>{title}</strong>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_dashboard_css()


# =========================================================
# SOURCE MODELS
# =========================================================
@dataclass(frozen=True)
class CsvSource:
    kind: str  # "path" or "bytes"
    label: str
    path: Optional[str] = None
    mtime: Optional[float] = None
    content: Optional[bytes] = None


# =========================================================
# BASIC HELPERS
# =========================================================
def normalize_text(val) -> str:
    if pd.isna(val):
        return ""
    val = str(val).strip()
    return "" if val.lower() == "nan" else val


def normalize_key(val) -> str:
    return normalize_text(val).lower()


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-_.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Robust numeric conversion for values like '1,234', '-', blanks, and strings."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": "0", "-": "0", "nan": "0", "None": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def format_number(x, decimals: int = 0) -> str:
    try:
        if pd.isna(x):
            return "-"
        value = float(x)
        if abs(value) < 1e-12:
            value = 0.0
        return f"{value:,.{decimals}f}"
    except Exception:
        return "-"


def format_percent(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.2f}%"
    except Exception:
        return "-"


def province_from_filename(filename: str) -> Optional[str]:
    """Infer province code from filename. Prefer exact token matches to avoid KP/KPC confusion."""
    stem = Path(filename).stem.upper()
    normalized = re.sub(r"[^A-Z0-9]+", "_", stem)
    tokens = [tok for tok in normalized.split("_") if tok]

    for province in sorted(PROVINCES, key=len, reverse=True):
        if normalized == province or province in tokens:
            return province

    for province in sorted(PROVINCES, key=len, reverse=True):
        if normalized.startswith(province + "_") or normalized.endswith("_" + province):
            return province

    return None


# =========================================================
# LOAD + PARSE CSV
# =========================================================
@st.cache_data(show_spinner=False)
def load_raw_csv_from_path(path_str: str, mtime: float) -> pd.DataFrame:
    # mtime is intentionally included so Streamlit invalidates cache when file changes.
    return pd.read_csv(path_str, header=None, dtype=str, keep_default_na=False)


@st.cache_data(show_spinner=False)
def load_raw_csv_from_bytes(content: bytes, label: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), header=None, dtype=str, keep_default_na=False)


def build_clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[0] < 2:
        raise ValueError("CSV must contain at least two header rows.")

    top = raw.iloc[0].fillna("").astype(str).str.strip().tolist()
    sub = raw.iloc[1].fillna("").astype(str).str.strip().tolist()

    columns = []
    current_group = None

    for i, (t, s) in enumerate(zip(top, sub)):
        if s in ["Region", "Cabin", "Consumer"]:
            columns.append(s)
            current_group = None
            continue

        if t in MONTH_FULL_LIST:
            current_group = t
        elif str(t).startswith("Accumulate for Year"):
            current_group = "ACC"

        if current_group in MONTH_FULL_LIST and s in ["Total kWh", "Sales kWh", "Losses"]:
            columns.append(f"{current_group}_{s}")
        elif current_group == "ACC" and s in ["Total kWh", "Sales kWh", "Losses"]:
            columns.append(f"ACC_{s}")
        else:
            columns.append(f"col_{i}")

    df = raw.iloc[2:].copy()
    df.columns = columns
    return df.reset_index(drop=True)


def collapse_duplicate_numeric_columns(df: pd.DataFrame, col_name: str) -> pd.Series:
    matched = df.loc[:, df.columns == col_name]
    if matched.shape[1] == 0:
        return pd.Series(0.0, index=df.index)
    return matched.apply(clean_numeric_series).sum(axis=1)


def build_consumer_series(df: pd.DataFrame) -> pd.Series:
    consumer_cols = df.loc[:, df.columns == "Consumer"]
    if consumer_cols.shape[1] == 0:
        return pd.Series(0.0, index=df.index)

    numeric_part = consumer_cols.apply(clean_numeric_series)
    if numeric_part.sum().sum() > 0:
        return numeric_part.sum(axis=1)

    text_part = consumer_cols.astype(str).replace("nan", "").apply(lambda c: c.str.strip())
    return (text_part != "").sum(axis=1).astype(float)


def classify_cabin_type_by_customers(customer_count: float) -> str:
    if customer_count == 1:
        return "Single"
    if customer_count > 1:
        return "Multiple"
    return "Unknown"


def get_cabin_counts(cabin_meta: pd.DataFrame) -> Dict[str, int]:
    if cabin_meta.empty:
        return {"All": 0, "Single": 0, "Multiple": 0, "Unknown": 0}
    return {
        "All": int(len(cabin_meta)),
        "Single": int((cabin_meta["type"] == "Single").sum()),
        "Multiple": int((cabin_meta["type"] == "Multiple").sum()),
        "Unknown": int((cabin_meta["type"] == "Unknown").sum()),
    }


@st.cache_data(show_spinner=False)
def prepare_raw_dataframe(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = build_clean_dataframe(raw)

    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].apply(normalize_text)
    else:
        df["Cabin"] = ""

    if "Region" in df.columns:
        df["Region"] = df["Region"].apply(normalize_text)
    else:
        df["Region"] = ""

    df["__cabin_key"] = df["Cabin"].apply(normalize_key)
    df["__display_name"] = df["Cabin"]

    for short_month in MONTHS:
        full_month = MONTH_MAP[short_month]
        df[f"__{short_month}_total"] = collapse_duplicate_numeric_columns(df, f"{full_month}_Total kWh")
        df[f"__{short_month}_sale"] = collapse_duplicate_numeric_columns(df, f"{full_month}_Sales kWh")

    df["__consumer_value"] = build_consumer_series(df)
    valid_df = df[df["__cabin_key"] != ""].copy()

    monthly_total_aggs = {f"{m}_total": (f"__{m}_total", "sum") for m in MONTHS}
    monthly_sale_aggs = {f"{m}_sale": (f"__{m}_sale", "sum") for m in MONTHS}

    if valid_df.empty:
        cabin_meta = pd.DataFrame(columns=[
            "__cabin_key", "Cabin", "display_name", "region", "customers", "rows", "type",
            *[f"{m}_total" for m in MONTHS],
            *[f"{m}_sale" for m in MONTHS],
        ])
    else:
        cabin_meta = valid_df.groupby("__cabin_key", as_index=False).agg(
            Cabin=("Cabin", "first"),
            display_name=("__display_name", "first"),
            region=("Region", "first"),
            customers=("__consumer_value", "sum"),
            rows=("__cabin_key", "size"),
            **monthly_total_aggs,
            **monthly_sale_aggs,
        )
        cabin_meta["customers"] = pd.to_numeric(cabin_meta["customers"], errors="coerce").fillna(0)
        cabin_meta["type"] = cabin_meta["customers"].apply(classify_cabin_type_by_customers)

    def sort_key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x).lower())

    if not cabin_meta.empty:
        cabin_meta = cabin_meta.sort_values(
            by="display_name",
            key=lambda s: s.map(sort_key),
        ).reset_index(drop=True)
        cabin_meta_indexed = cabin_meta.set_index("__cabin_key", drop=False)
    else:
        cabin_meta_indexed = cabin_meta

    return {
        "df": df,
        "cabin_meta": cabin_meta,
        "cabin_meta_indexed": cabin_meta_indexed,
        "counts": pd.DataFrame([get_cabin_counts(cabin_meta)]),
    }


def prepare_source(source: CsvSource) -> Dict[str, pd.DataFrame]:
    if source.kind == "path":
        raw = load_raw_csv_from_path(source.path or "", source.mtime or 0.0)
    elif source.kind == "bytes":
        raw = load_raw_csv_from_bytes(source.content or b"", source.label)
    else:
        raise ValueError(f"Unsupported source kind: {source.kind}")
    return prepare_raw_dataframe(raw)


def build_static_sources() -> Dict[int, Dict[str, CsvSource]]:
    sources: Dict[int, Dict[str, CsvSource]] = {2024: {}, 2025: {}}
    for year in HISTORICAL_YEARS:
        folder = DATA_ROOT / str(year)
        for province in PROVINCES:
            path = folder / f"{province}.csv"
            if path.exists():
                sources[year][province] = CsvSource(
                    kind="path",
                    label=f"{year}/{province}.csv",
                    path=str(path),
                    mtime=path.stat().st_mtime,
                )
    return sources


def build_uploaded_2026_sources(uploaded_files) -> Tuple[Dict[str, CsvSource], list[str], list[str]]:
    sources: Dict[str, CsvSource] = {}
    rejected: list[str] = []
    duplicates: list[str] = []

    for uploaded in uploaded_files or []:
        province = province_from_filename(uploaded.name)
        if province is None:
            rejected.append(uploaded.name)
            continue
        if province in sources:
            duplicates.append(uploaded.name)
        sources[province] = CsvSource(
            kind="bytes",
            label=uploaded.name,
            content=uploaded.getvalue(),
        )

    return sources, rejected, duplicates


def get_year_source(year_sources: Dict[int, Dict[str, CsvSource]], year: int, province: str) -> Optional[CsvSource]:
    return year_sources.get(year, {}).get(province)


# =========================================================
# SUMMARY + RANKING LOGIC
# =========================================================
def build_summary_table_from_row(row: pd.Series) -> pd.DataFrame:
    monthly_sale, monthly_total, monthly_diff, monthly_loss_pct = [], [], [], []

    for short_month in MONTHS:
        total_val = float(row.get(f"{short_month}_total", 0.0))
        sale_val = float(row.get(f"{short_month}_sale", 0.0))
        diff_val = total_val - sale_val
        loss_pct = 0.0 if total_val == 0 else (1 - sale_val / total_val) * 100

        monthly_sale.append(sale_val)
        monthly_total.append(total_val)
        monthly_diff.append(diff_val)
        monthly_loss_pct.append(loss_pct)

    sale_acc = sum(monthly_sale)
    total_acc = sum(monthly_total)
    diff_acc = total_acc - sale_acc
    loss_acc = 0.0 if total_acc == 0 else (1 - sale_acc / total_acc) * 100

    return pd.DataFrame([
        [1, "Sale", "kWh", *monthly_sale, sale_acc, sale_acc / 12],
        [2, "Total", "kWh", *monthly_total, total_acc, total_acc / 12],
        [3, "total - sale", "kWh", *monthly_diff, diff_acc, diff_acc / 12],
        [4, "losses", "%", *monthly_loss_pct, loss_acc, sum(monthly_loss_pct) / 12],
    ], columns=["No", "Description", "Unit", *MONTHS, "Accumulate", "Average"])


def format_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    # Pandas in newer Streamlit Cloud environments is stricter about assigning
    # strings such as "1,234" or "2.50%" into numeric columns. Keep the raw
    # summary table numeric for calculations, but cast only the display copy
    # to object before writing formatted strings.
    display_df = summary_df.copy().astype(object)
    value_cols = MONTHS + ["Accumulate", "Average"]

    for idx in display_df.index:
        desc = str(display_df.at[idx, "Description"]).strip()

        if desc in ["Sale", "Total", "total - sale"]:
            for col in value_cols:
                display_df.at[idx, col] = format_number(display_df.at[idx, col], 0)

        elif desc == "losses":
            for col in value_cols:
                display_df.at[idx, col] = format_percent(display_df.at[idx, col])

    return display_df.astype(str)


def extract_monthly_losses_from_row(row: pd.Series) -> list[float]:
    values = []
    for short_month in MONTHS:
        total_val = float(row.get(f"{short_month}_total", 0.0))
        sale_val = float(row.get(f"{short_month}_sale", 0.0))
        values.append(0.0 if total_val == 0 else (1 - sale_val / total_val) * 100)
    return values


def get_row_from_meta(meta_indexed: pd.DataFrame, cabin_key: str) -> Optional[pd.Series]:
    if meta_indexed.empty or cabin_key not in meta_indexed.index:
        return None
    row = meta_indexed.loc[cabin_key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row




def resolve_cabin_text_input(cabin_options_df: pd.DataFrame, cabin_query: str) -> Tuple[Optional[str], str, pd.DataFrame]:
    """Resolve a typed cabin query into one cabin key.

    Matching order:
    1) exact cabin key/display name match, case-insensitive
    2) partial display-name match
    Returns: selected_key, message, matches_df
    """
    query = normalize_text(cabin_query)
    if query == "":
        return None, "Type a cabin number/name to open the summary.", pd.DataFrame()

    if cabin_options_df.empty:
        return None, "No cabins are available under the current filter.", pd.DataFrame()

    query_key = normalize_key(query)
    working = cabin_options_df.copy()
    working["__display_key"] = working["display_name"].apply(normalize_key)

    exact = working[(working["__cabin_key"] == query_key) | (working["__display_key"] == query_key)]
    if not exact.empty:
        row = exact.iloc[0]
        return row["__cabin_key"], f"Opened exact match: Cabin {row['display_name']}", pd.DataFrame()

    partial = working[working["__display_key"].str.contains(re.escape(query_key), na=False)].copy()
    if partial.empty:
        return None, f"No cabin found for: {query}", pd.DataFrame()

    if len(partial) == 1:
        row = partial.iloc[0]
        return row["__cabin_key"], f"Opened partial match: Cabin {row['display_name']}", pd.DataFrame()

    preview_cols = ["display_name", "type", "customers", "rows"]
    matches_df = partial[preview_cols].head(30).rename(columns={
        "display_name": "Cabin",
        "type": "Type",
        "customers": "Customers",
        "rows": "Matched Rows",
    })
    return None, (
        f"Your input matches {len(partial)} cabins. Type the exact cabin number/name. "
        "The first 30 matches are shown below."
    ), matches_df


def build_single_cabin_gap_ranking(cabin_meta: pd.DataFrame, month: str) -> pd.DataFrame:
    if cabin_meta.empty:
        return pd.DataFrame()

    meta = cabin_meta[cabin_meta["type"] == "Single"].copy()
    if meta.empty:
        return pd.DataFrame()

    total_col = f"{month}_total"
    sale_col = f"{month}_sale"
    meta["rank_total"] = pd.to_numeric(meta[total_col], errors="coerce").fillna(0)
    meta["rank_sale"] = pd.to_numeric(meta[sale_col], errors="coerce").fillna(0)
    meta["rank_gap"] = meta["rank_total"] - meta["rank_sale"]
    meta["rank_loss_pct"] = 0.0
    mask = meta["rank_total"] != 0
    meta.loc[mask, "rank_loss_pct"] = (1 - meta.loc[mask, "rank_sale"] / meta.loc[mask, "rank_total"]) * 100

    meta = meta.sort_values(
        by=["rank_gap", "rank_loss_pct", "rank_total"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    meta.insert(0, "Rank", range(1, len(meta) + 1))
    return meta


def make_ranking_display_df(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df
    out = ranking_df[["Rank", "display_name", "rank_total", "rank_sale", "rank_gap", "rank_loss_pct", "customers"]].copy()
    out = out.rename(columns={
        "display_name": "Cabin",
        "rank_total": "Total kWh",
        "rank_sale": "Sale kWh",
        "rank_gap": "Total - Sale",
        "rank_loss_pct": "Loss %",
        "customers": "Customers",
    })
    out["Total kWh"] = out["Total kWh"].map(lambda x: format_number(x, 0))
    out["Sale kWh"] = out["Sale kWh"].map(lambda x: format_number(x, 0))
    out["Total - Sale"] = out["Total - Sale"].map(lambda x: format_number(x, 0))
    out["Loss %"] = out["Loss %"].map(format_percent)
    out["Customers"] = out["Customers"].map(lambda x: format_number(x, 0))
    return out


# =========================================================
# CHARTS
# =========================================================
def build_sale_total_chart(summary_df: pd.DataFrame, year: int = 2026) -> go.Figure:
    sale = summary_df.loc[summary_df["Description"] == "Sale", MONTHS].iloc[0].astype(float).tolist()
    total = summary_df.loc[summary_df["Description"] == "Total", MONTHS].iloc[0].astype(float).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MONTHS, y=sale, mode="lines+markers", name=f"Sale ({year})",
        line=dict(shape="spline", smoothing=1.1, width=3),
        hovertemplate="<b>%{x}</b><br>Sale: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=MONTHS, y=total, mode="lines+markers", name=f"Total ({year})",
        line=dict(shape="spline", smoothing=1.1, width=3),
        hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"Sale vs Total by Month ({year})",
        height=400,
        hovermode="x unified",
        margin=dict(l=30, r=20, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Month"),
        yaxis=dict(title="kWh", separatethousands=True),
    )
    return fig


def build_multi_year_loss_chart(loss_by_year: Dict[int, list[float]]) -> go.Figure:
    fig = go.Figure()
    for year in sorted(loss_by_year):
        losses = loss_by_year[year]
        fig.add_trace(go.Scatter(
            x=MONTHS,
            y=losses,
            mode="lines+markers+text",
            name=f"Loss % ({year})",
            text=[f"{v:.2f}%" for v in losses],
            textposition="top center",
            line=dict(shape="spline", smoothing=1.1, width=3),
            hovertemplate=f"<b>%{{x}}</b><br>Loss {year}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="Loss % by Month",
        height=470,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Loss %"),
    )
    return fig


# =========================================================
# PNG EXPORT HELPERS
# =========================================================
@lru_cache(maxsize=32)
def get_pil_font(size: int = 18, bold: bool = False):
    if bold:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "arialbd.ttf",
        ]
    else:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "arial.ttf",
        ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_centered_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    text = str(text)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=font, fill=fill)


def draw_label_box(draw, x, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]
    pad = 3
    box = [x - pad, y - pad, x + label_w + pad, y + label_h + pad]
    draw.rounded_rectangle(box, radius=4, fill="white", outline="#e5e7eb")
    draw.text((x, y), text, font=font, fill=fill)


def catmull_rom_spline(points, samples_per_segment=18):
    if len(points) < 2:
        return points
    extended = [points[0]] + points + [points[-1]]
    curve = []
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            curve.append((x, y))
    curve.append(points[-1])
    return curve


def summary_row_colors(desc: str):
    d = str(desc).lower()
    # Only Gap / total - sale keeps row color. Sale, Total, and Loss % stay plain.
    if d == "total - sale" or d == "gap":
        return "#fff7ed", "#c2410c"
    return "white", "#111827"


def make_loss_curve_png_image(loss_by_year: Dict[int, list[float]], title: str, subtitle: Optional[str] = None) -> Image.Image:
    width, height = 1345, 500
    margin_l, margin_r, margin_t, margin_b = 80, 35, 115, 62
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_pil_font(24, bold=True)
    font_subtitle = get_pil_font(15)
    font_axis = get_pil_font(13)
    font_legend = get_pil_font(14, bold=True)
    font_label = get_pil_font(12, bold=True)

    draw.text((18, 16), title, font=font_title, fill="#0f172a")
    if subtitle:
        draw.text((18, 50), subtitle[:180], font=font_subtitle, fill="#475569")

    plot_x1, plot_y1 = margin_l, margin_t
    plot_x2, plot_y2 = width - margin_r, height - margin_b

    all_values = [v for values in loss_by_year.values() for v in values]
    if not all_values:
        all_values = [0.0]
    y_min = min(0, math.floor(min(all_values) / 5) * 5)
    y_max = max(5, math.ceil(max(all_values) / 5) * 5)
    if y_max == y_min:
        y_max = y_min + 5
    y_padding = (y_max - y_min) * 0.16
    y_min -= y_padding
    y_max += y_padding

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y = plot_y2 - (y_val - y_min) / (y_max - y_min) * (plot_y2 - plot_y1)
        draw.line([(plot_x1, y), (plot_x2, y)], fill="#e5e7eb", width=1)
        draw.text((18, y - 8), f"{y_val:.1f}%", font=font_axis, fill="#64748b")

    draw.line([(plot_x1, plot_y1), (plot_x1, plot_y2)], fill="#334155", width=2)
    draw.line([(plot_x1, plot_y2), (plot_x2, plot_y2)], fill="#334155", width=2)

    x_positions = []
    for idx, month in enumerate(MONTHS):
        x = plot_x1 + idx * (plot_x2 - plot_x1) / (len(MONTHS) - 1)
        x_positions.append(x)
        draw.line([(x, plot_y2), (x, plot_y2 + 5)], fill="#334155", width=1)
        draw_centered_text(draw, (x - 25, plot_y2 + 10, x + 25, plot_y2 + 35), month, font_axis, "#334155")

    year_styles = {
        2024: {"color": "#f59e0b", "label_offset": -24},
        2025: {"color": "#8b5cf6", "label_offset": 18},
        2026: {"color": "#ef4444", "label_offset": -42},
    }

    legend_x, legend_y = width - 420, 22
    for idx, year in enumerate(sorted(loss_by_year)):
        color = year_styles.get(year, {"color": "#334155"})["color"]
        ly = legend_y + idx * 24
        draw.line([(legend_x, ly + 8), (legend_x + 36, ly + 8)], fill=color, width=4)
        draw.ellipse([legend_x + 14, ly + 2, legend_x + 24, ly + 12], fill=color)
        draw.text((legend_x + 48, ly), f"Loss % ({year})", font=font_legend, fill="#0f172a")

    for year in sorted(loss_by_year):
        color = year_styles.get(year, {"color": "#334155", "label_offset": -24})["color"]
        label_offset = year_styles.get(year, {"label_offset": -24})["label_offset"]
        values = loss_by_year[year]
        points = []
        for idx, val in enumerate(values):
            x = x_positions[idx]
            y = plot_y2 - (val - y_min) / (y_max - y_min) * (plot_y2 - plot_y1)
            points.append((x, y))
        curve = catmull_rom_spline(points, samples_per_segment=14)
        if len(curve) > 1:
            draw.line(curve, fill=color, width=4)
        for idx, (x, y) in enumerate(points):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline="white", width=2)
            label = f"{values[idx]:.2f}%"
            bbox = draw.textbbox((0, 0), label, font=font_label)
            label_w = bbox[2] - bbox[0]
            label_y = max(plot_y1 - 34, min(y + label_offset, plot_y2 + 8))
            label_x = max(plot_x1 - 20, min(x - label_w / 2, plot_x2 - label_w + 20))
            draw_label_box(draw, label_x, label_y, label, font_label, color)

    return img


def make_summary_table_png_image(display_df: pd.DataFrame, title: str, subtitle: Optional[str] = None) -> Image.Image:
    font_title = get_pil_font(24, bold=True)
    font_subtitle = get_pil_font(15)
    font_header = get_pil_font(13, bold=True)
    font_cell = get_pil_font(13, bold=True)

    headers = display_df.columns.tolist()
    data = [headers] + display_df.astype(str).values.tolist()
    col_widths = [55, 135, 70] + [78] * 12 + [115, 105]
    row_h = 38
    title_h = 80 if subtitle else 56
    width = sum(col_widths) + 2
    height = title_h + row_h * len(data) + 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, title_h], fill="#ffffff")
    draw.text((12, 14), title, font=font_title, fill="#0f172a")
    if subtitle:
        draw.text((12, 48), subtitle[:160], font=font_subtitle, fill="#475569")

    y = title_h
    for r_idx, row in enumerate(data):
        x = 0
        if r_idx == 0:
            bg, fg, font = "#0f172a", "white", font_header
        else:
            bg, fg = summary_row_colors(row[1])
            font = font_cell
        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            draw.rectangle([x, y, x + w, y + row_h], fill=bg, outline="#e5e7eb")
            text = str(cell)
            if c_idx >= 3 and len(text) > 16:
                text = text[:15] + "…"
            draw_centered_text(draw, (x + 3, y + 3, x + w - 3, y + row_h - 3), text, font, fg)
            x += w
        y += row_h
    return img


def make_missing_year_image(year: int, province: str, cabin_name: str, rank_no: int, ranking_month: str) -> Image.Image:
    missing_df = pd.DataFrame([
        ["-", f"No {year} data found", "-", *["-"] * 12, "-", "-"]
    ], columns=["No", "Description", "Unit", *MONTHS, "Accumulate", "Average"])
    return make_summary_table_png_image(
        missing_df,
        title=f"Summary Table ({year})",
        subtitle=f"Rank {rank_no:03d} | Province {province} | Cabin {cabin_name} | Ranking month: {ranking_month} {LATEST_YEAR}",
    )


def image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    """Save PNG with DPI metadata.

    Important: DPI metadata alone does not create sharp print output.
    The image canvas must also be large enough. That is why the app now
    exports high-resolution A4 PNG pages instead of one long low-resolution PNG.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, dpi=(dpi, dpi))
    return buf.getvalue()


def make_combined_summary_png_bytes(
    province: str,
    ranking_month: str,
    cabin_key: str,
    rank_row: pd.Series,
    meta_by_year: Dict[int, pd.DataFrame],
) -> bytes:
    cabin_name = rank_row["display_name"]
    rank_no = int(rank_row["Rank"])

    loss_by_year = {}
    for year, meta in meta_by_year.items():
        row = get_row_from_meta(meta, cabin_key)
        if row is not None:
            loss_by_year[year] = extract_monthly_losses_from_row(row)

    subtitle = (
        f"Province: {province} | Cabin: {cabin_name} | Ranking Month: {ranking_month} {LATEST_YEAR} | "
        f"Total - Sale: {format_number(rank_row['rank_gap'], 0)} kWh | Loss: {format_percent(rank_row['rank_loss_pct'])}"
    )
    chart_img = make_loss_curve_png_image(loss_by_year, "Loss % Curve Chart (2024, 2025, 2026)", subtitle)

    table_images = []
    for year in [2026, 2025, 2024]:
        row = get_row_from_meta(meta_by_year.get(year, pd.DataFrame()), cabin_key)
        if row is None:
            img = make_missing_year_image(year, province, cabin_name, rank_no, ranking_month)
        else:
            summary_df = build_summary_table_from_row(row)
            display_df = format_summary_for_display(summary_df)
            table_subtitle = f"Rank {rank_no:03d} | Province {province} | Cabin {cabin_name} | Ranking Month: {ranking_month} {LATEST_YEAR}"
            img = make_summary_table_png_image(display_df, title=f"Summary Table ({year})", subtitle=table_subtitle)
        table_images.append(img)

    margin, gap, header_h = 30, 26, 135
    width = max(chart_img.width, max(img.width for img in table_images)) + margin * 2
    height = header_h + chart_img.height + gap + sum(img.height for img in table_images) + gap * len(table_images) + margin
    combined = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(combined)

    font_title = get_pil_font(30, bold=True)
    font_body = get_pil_font(17)
    draw.text((margin, 22), f"Single Cabin Ranking Summary | Rank {rank_no:03d}", font=font_title, fill="#0f172a")
    context_lines = [
        f"Province: {province} | Cabin: {cabin_name} | Cabin Type: Single",
        f"Ranking Month: {ranking_month} {LATEST_YEAR}",
        f"Total kWh: {format_number(rank_row['rank_total'], 0)} | Sale kWh: {format_number(rank_row['rank_sale'], 0)} | "
        f"Total - Sale: {format_number(rank_row['rank_gap'], 0)} | Loss: {format_percent(rank_row['rank_loss_pct'])}",
    ]
    y_text = 64
    for line in context_lines:
        draw.text((margin, y_text), line, font=font_body, fill="#334155")
        y_text += 22

    y = header_h
    combined.paste(chart_img, (margin, y))
    y += chart_img.height + gap
    for img in table_images:
        combined.paste(img, (margin, y))
        y += img.height + gap

    return image_to_png_bytes(combined)


def build_export_zip_bytes(
    province: str,
    ranking_month: str,
    visible_ranking_df: pd.DataFrame,
    meta_by_year: Dict[int, pd.DataFrame],
) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for _, rank_row in visible_ranking_df.iterrows():
            cabin_key = rank_row["__cabin_key"]
            cabin_name = rank_row["display_name"]
            file_name = safe_filename(
                f"{int(rank_row['Rank']):03d}_{province}_{ranking_month}_Cabin_{cabin_name}_Chart_and_3_Summary_Tables.png"
            )
            png_bytes = make_combined_summary_png_bytes(province, ranking_month, cabin_key, rank_row, meta_by_year)
            zipf.writestr(file_name, png_bytes)
    return zip_buffer.getvalue()




# =========================================================
# LIGHTWEIGHT REPORT HELPERS
# =========================================================
def get_summary_and_loss_for_cabin(
    year_sources: Dict[int, Dict[str, CsvSource]],
    province: str,
    selected_cabin_key: str,
    data_2026: Dict[str, pd.DataFrame],
    selected_row_2026: pd.Series,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, list[float]], Dict[int, pd.DataFrame], Dict[int, str]]:
    """Load only the selected province/year data needed for the selected cabin."""
    summary_by_year: Dict[int, pd.DataFrame] = {}
    loss_by_year: Dict[int, list[float]] = {}
    meta_by_year: Dict[int, pd.DataFrame] = {}
    status_by_year: Dict[int, str] = {}

    meta_by_year[2026] = data_2026["cabin_meta_indexed"]
    summary_by_year[2026] = build_summary_table_from_row(selected_row_2026)
    loss_by_year[2026] = extract_monthly_losses_from_row(selected_row_2026)
    status_by_year[2026] = "OK"

    for year in [2025, 2024]:
        source = get_year_source(year_sources, year, province)
        if source is None:
            status_by_year[year] = f"No {year} CSV found for {province} in repository."
            meta_by_year[year] = pd.DataFrame()
            continue

        try:
            data_year = prepare_source(source)
            meta_year = data_year["cabin_meta_indexed"]
            meta_by_year[year] = meta_year
            row_year = get_row_from_meta(meta_year, selected_cabin_key)
            if row_year is None:
                status_by_year[year] = f"Cabin not found in {year}."
            else:
                summary_by_year[year] = build_summary_table_from_row(row_year)
                loss_by_year[year] = extract_monthly_losses_from_row(row_year)
                status_by_year[year] = "OK"
        except Exception as exc:
            status_by_year[year] = f"Could not read {year}: {exc}"
            meta_by_year[year] = pd.DataFrame()

    return summary_by_year, loss_by_year, meta_by_year, status_by_year


def get_value_from_summary(summary_df: pd.DataFrame, description: str, column: str) -> float:
    try:
        return float(summary_df.loc[summary_df["Description"] == description, column].iloc[0])
    except Exception:
        return 0.0


def build_yearly_kpi_table(summary_by_year: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year in sorted(summary_by_year.keys()):
        summary = summary_by_year[year]
        total_acc = get_value_from_summary(summary, "Total", "Accumulate")
        sale_acc = get_value_from_summary(summary, "Sale", "Accumulate")
        gap_acc = get_value_from_summary(summary, "total - sale", "Accumulate")
        weighted_loss = get_value_from_summary(summary, "losses", "Accumulate")
        avg_loss = get_value_from_summary(summary, "losses", "Average")

        rows.append({
            "Year": year,
            "Total kWh": total_acc,
            "Sale kWh": sale_acc,
            "Total - Sale": gap_acc,
            "Weighted Loss %": weighted_loss,
            "Avg Monthly Loss %": avg_loss,
        })

    return pd.DataFrame(rows)


def format_yearly_kpi_table(kpi_df: pd.DataFrame) -> pd.DataFrame:
    if kpi_df.empty:
        return kpi_df
    out = kpi_df.copy().astype(object)
    for col in ["Total kWh", "Sale kWh", "Total - Sale"]:
        out[col] = out[col].map(lambda x: format_number(x, 0))
    for col in ["Weighted Loss %", "Avg Monthly Loss %"]:
        out[col] = out[col].map(format_percent)
    return out.astype(str)


def build_loss_comparison_table(
    summary_by_year: Dict[int, pd.DataFrame],
    loss_by_year: Dict[int, list[float]],
) -> pd.DataFrame:
    rows = []
    for idx, month in enumerate(MONTHS):
        row = {"Month": month}
        for year in ALL_YEARS:
            if year in loss_by_year:
                row[str(year)] = loss_by_year[year][idx]
            else:
                row[str(year)] = None
        rows.append(row)

    weighted_row = {"Month": "Weighted yearly loss"}
    average_row = {"Month": "Average monthly loss"}
    for year in ALL_YEARS:
        if year in summary_by_year:
            weighted_row[str(year)] = get_value_from_summary(summary_by_year[year], "losses", "Accumulate")
            average_row[str(year)] = get_value_from_summary(summary_by_year[year], "losses", "Average")
        else:
            weighted_row[str(year)] = None
            average_row[str(year)] = None

    rows.extend([weighted_row, average_row])
    return pd.DataFrame(rows)


def format_loss_comparison_table(loss_df: pd.DataFrame) -> pd.DataFrame:
    if loss_df.empty:
        return loss_df
    out = loss_df.copy().astype(object)
    for year in [str(y) for y in ALL_YEARS]:
        if year in out.columns:
            out[year] = out[year].map(lambda x: "-" if pd.isna(x) else format_percent(x))
    return out.astype(str)


def build_all_year_summary_csv(summary_by_year: Dict[int, pd.DataFrame]) -> bytes:
    frames = []
    for year in sorted(summary_by_year.keys(), reverse=True):
        temp = summary_by_year[year].copy()
        temp.insert(0, "Year", year)
        frames.append(temp)
    if not frames:
        return b""
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False).encode("utf-8-sig")


def build_ranking_csv_bytes(ranking_df: pd.DataFrame) -> bytes:
    if ranking_df.empty:
        return b""
    export_df = ranking_df[[
        "Rank", "display_name", "rank_total", "rank_sale", "rank_gap", "rank_loss_pct", "customers"
    ]].rename(columns={
        "display_name": "Cabin",
        "rank_total": "Total kWh",
        "rank_sale": "Sale kWh",
        "rank_gap": "Total - Sale",
        "rank_loss_pct": "Loss %",
        "customers": "Customers",
    })
    return export_df.to_csv(index=False).encode("utf-8-sig")


def make_generic_table_png_image(
    display_df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    col_widths: Optional[list[int]] = None,
    row_h: int = 40,
) -> Image.Image:
    font_title = get_pil_font(24, bold=True)
    font_subtitle = get_pil_font(15)
    font_header = get_pil_font(13, bold=True)
    font_cell = get_pil_font(13, bold=False)

    headers = display_df.columns.astype(str).tolist()
    data = [headers] + display_df.astype(str).values.tolist()

    if col_widths is None:
        col_widths = []
        for col_idx, header in enumerate(headers):
            max_len = len(str(header))
            for row in display_df.astype(str).values.tolist():
                max_len = max(max_len, len(str(row[col_idx])))
            col_widths.append(max(105, min(210, max_len * 9 + 28)))

    title_h = 82 if subtitle else 58
    width = sum(col_widths) + 2
    height = title_h + row_h * len(data) + 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, title_h], fill="#ffffff")
    draw.text((12, 14), title, font=font_title, fill="#0f172a")
    if subtitle:
        draw.text((12, 48), subtitle[:170], font=font_subtitle, fill="#475569")

    y = title_h
    for r_idx, row in enumerate(data):
        x = 0
        if r_idx == 0:
            bg, fg, font = "#0f172a", "white", font_header
        else:
            bg, fg, font = "#ffffff", "#111827", font_cell
            if str(row[0]).lower().startswith("weighted") or str(row[0]).lower().startswith("average"):
                bg, fg, font = "#f8fafc", "#0f172a", get_pil_font(13, bold=True)

        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            draw.rectangle([x, y, x + w, y + row_h], fill=bg, outline="#e5e7eb")
            text = str(cell)
            if len(text) > 24:
                text = text[:23] + "…"
            draw_centered_text(draw, (x + 4, y + 4, x + w - 4, y + row_h - 4), text, font, fg)
            x += w
        y += row_h

    return img


def make_printable_selected_report_png_bytes(
    province: str,
    cabin_name: str,
    cabin_type: str,
    ranking_month: str,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_by_year: Dict[int, list[float]],
    yearly_kpi_df: pd.DataFrame,
    loss_compare_df: pd.DataFrame,
) -> bytes:
    subtitle = f"Province: {province} | Cabin: {cabin_name} | Type: {cabin_type} | Ranking month: {ranking_month} {LATEST_YEAR}"

    kpi_img = make_generic_table_png_image(
        format_yearly_kpi_table(yearly_kpi_df),
        title="Yearly KPI Comparison",
        subtitle=subtitle,
        col_widths=[85, 150, 150, 150, 150, 160],
        row_h=42,
    )

    chart_img = make_loss_curve_png_image(
        loss_by_year=loss_by_year,
        title="Loss % Trend by Month",
        subtitle=subtitle,
    )

    loss_compare_img = make_generic_table_png_image(
        format_loss_comparison_table(loss_compare_df),
        title="Monthly Loss % Comparison",
        subtitle="This table compares loss percentage month-by-month across available years.",
        col_widths=[190, 130, 130, 130],
        row_h=38,
    )

    table_images = []
    for year in [2026, 2025, 2024]:
        if year in summary_by_year:
            table_images.append(
                make_summary_table_png_image(
                    format_summary_for_display(summary_by_year[year]),
                    title=f"Summary Table ({year})",
                    subtitle=subtitle,
                )
            )

    margin = 36
    gap = 24
    header_h = 120
    report_w = max([kpi_img.width, loss_compare_img.width, *[img.width for img in table_images]]) + margin * 2
    report_h = header_h + kpi_img.height + gap + loss_compare_img.height + gap
    report_h += sum(img.height for img in table_images) + gap * len(table_images) + margin

    report = Image.new("RGB", (report_w, report_h), "white")
    draw = ImageDraw.Draw(report)

    title_font = get_pil_font(32, bold=True)
    body_font = get_pil_font(17)
    small_font = get_pil_font(14)

    draw.text((margin, 24), "EDC Cabin Loss Printable Report", font=title_font, fill="#0f172a")
    draw.text((margin, 66), subtitle, font=body_font, fill="#334155")
    draw.text(
        (margin, 92),
        "Use this report for one selected cabin. Batch PNG export is intentionally avoided to keep Streamlit Cloud stable.",
        font=small_font,
        fill="#64748b",
    )

    y = header_h
    for img in [kpi_img, loss_compare_img, *table_images]:
        report.paste(img, (margin, y))
        y += img.height + gap

    return image_to_png_bytes(report)


def build_selected_report_zip_bytes(
    province: str,
    cabin_name: str,
    ranking_month: str,
    report_png_bytes: bytes,
    visible_ranking_df: pd.DataFrame,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_compare_df: pd.DataFrame,
    yearly_kpi_df: pd.DataFrame,
) -> bytes:
    zip_buffer = io.BytesIO()
    safe_cabin = safe_filename(cabin_name)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_{ranking_month}_{LATEST_YEAR}_Printable_Report.png"),
            report_png_bytes,
        )
        zipf.writestr(
            safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_Visible_Ranking.csv"),
            build_ranking_csv_bytes(visible_ranking_df),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Summary_2024_2025_2026.csv"),
            build_all_year_summary_csv(summary_by_year),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Loss_Comparison.csv"),
            loss_compare_df.to_csv(index=False).encode("utf-8-sig"),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Yearly_KPI.csv"),
            yearly_kpi_df.to_csv(index=False).encode("utf-8-sig"),
        )
    return zip_buffer.getvalue()


# =========================================================
# PRINT-READY PDF + HIGH-RESOLUTION PNG EXPORT HELPERS
# =========================================================
PRINT_DPI = 300
A4_LANDSCAPE_PX = (3508, 2480)  # A4 at 300 DPI
A4_PORTRAIT_PX = (2480, 3508)   # A4 at 300 DPI


def make_loss_curve_png_image_sized(
    loss_by_year: Dict[int, list[float]],
    title: str,
    subtitle: Optional[str] = None,
    width: int = 3200,
    height: int = 1250,
) -> Image.Image:
    """High-resolution loss chart for A4 landscape print/PDF export.

    Design goal: readable labels and simple comparison. The chart is exported
    large, then scaled into the PDF, so point labels remain clear when printed.
    """
    margin_l = 190
    margin_r = 100
    margin_t = 235
    margin_b = 150

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_pil_font(58, bold=True)
    font_subtitle = get_pil_font(30)
    font_axis = get_pil_font(30)
    font_legend = get_pil_font(32, bold=True)
    font_label = get_pil_font(38, bold=True)

    draw.text((42, 36), title, font=font_title, fill="#0f172a")
    if subtitle:
        draw.text((42, 118), subtitle[:185], font=font_subtitle, fill="#475569")

    plot_x1, plot_y1 = margin_l, margin_t
    plot_x2, plot_y2 = width - margin_r, height - margin_b

    all_values = [v for values in loss_by_year.values() for v in values]
    if not all_values:
        all_values = [0.0]

    y_min = min(0, math.floor(min(all_values) / 5) * 5)
    y_max = max(5, math.ceil(max(all_values) / 5) * 5)
    if y_max == y_min:
        y_max = y_min + 5

    y_padding = (y_max - y_min) * 0.20
    y_min -= y_padding
    y_max += y_padding

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y = plot_y2 - (y_val - y_min) / (y_max - y_min) * (plot_y2 - plot_y1)
        draw.line([(plot_x1, y), (plot_x2, y)], fill="#e5e7eb", width=3)
        draw.text((42, y - 18), f"{y_val:.1f}%", font=font_axis, fill="#64748b")

    draw.line([(plot_x1, plot_y1), (plot_x1, plot_y2)], fill="#334155", width=5)
    draw.line([(plot_x1, plot_y2), (plot_x2, plot_y2)], fill="#334155", width=5)

    x_positions = []
    for idx, month in enumerate(MONTHS):
        x = plot_x1 + idx * (plot_x2 - plot_x1) / (len(MONTHS) - 1)
        x_positions.append(x)
        draw.line([(x, plot_y2), (x, plot_y2 + 12)], fill="#334155", width=3)
        draw_centered_text(draw, (x - 64, plot_y2 + 28, x + 64, plot_y2 + 88), month, font_axis, "#334155")

    year_styles = {
        2024: {"color": "#f59e0b", "label_offset": -80},
        2025: {"color": "#8b5cf6", "label_offset": 48},
        2026: {"color": "#ef4444", "label_offset": -138},
    }

    legend_x, legend_y = width - 930, 46
    for idx, year in enumerate(sorted(loss_by_year)):
        color = year_styles.get(year, {"color": "#334155"})["color"]
        ly = legend_y + idx * 62
        draw.line([(legend_x, ly + 22), (legend_x + 90, ly + 22)], fill=color, width=9)
        draw.ellipse([legend_x + 34, ly + 9, legend_x + 56, ly + 31], fill=color)
        draw.text((legend_x + 120, ly), f"Loss % ({year})", font=font_legend, fill="#0f172a")

    for year in sorted(loss_by_year):
        style = year_styles.get(year, {"color": "#334155", "label_offset": -80})
        color = style["color"]
        label_offset = style["label_offset"]
        values = loss_by_year[year]
        points = []
        for idx, val in enumerate(values):
            x = x_positions[idx]
            y = plot_y2 - (val - y_min) / (y_max - y_min) * (plot_y2 - plot_y1)
            points.append((x, y))
        curve = catmull_rom_spline(points, samples_per_segment=18)
        if len(curve) > 1:
            draw.line(curve, fill=color, width=9)
        for idx, (x, y) in enumerate(points):
            draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=color, outline="white", width=5)
            label = f"{values[idx]:.2f}%"
            bbox = draw.textbbox((0, 0), label, font=font_label)
            label_w = bbox[2] - bbox[0]
            label_y = max(plot_y1 - 100, min(y + label_offset, plot_y2 + 24))
            label_x = max(plot_x1 - 60, min(x - label_w / 2, plot_x2 - label_w + 60))
            draw_label_box(draw, label_x, label_y, label, font_label, color)

    return img

def draw_table_on_image(
    draw: ImageDraw.ImageDraw,
    display_df: pd.DataFrame,
    x: int,
    y: int,
    width: int,
    row_h: int,
    header_font,
    cell_font,
    col_weights: Optional[list[float]] = None,
    first_col_bold: bool = True,
) -> int:
    """Draw a dataframe table and return the y position after the table."""
    headers = display_df.columns.astype(str).tolist()
    rows = [headers] + display_df.astype(str).values.tolist()
    n_cols = len(headers)

    if col_weights is None or len(col_weights) != n_cols:
        col_weights = [1.0] * n_cols
    weight_sum = sum(col_weights)
    col_widths = [int(width * w / weight_sum) for w in col_weights]
    col_widths[-1] += width - sum(col_widths)

    for r_idx, row in enumerate(rows):
        cx = x
        is_header = r_idx == 0
        row_first_value = str(row[0]).lower() if len(row) else ""
        is_total_row = row_first_value.startswith("weighted") or row_first_value.startswith("average")

        if is_header:
            bg, fg, font = "#0f172a", "white", header_font
        elif is_total_row:
            bg, fg, font = "#f8fafc", "#0f172a", get_pil_font(max(18, cell_font.size), bold=True) if hasattr(cell_font, "size") else cell_font
        else:
            bg, fg, font = "white", "#111827", cell_font

        # Color summary rows by metric/description.
        # Only Gap is highlighted; Sale, Total, and Loss % stay plain.
        if not is_header:
            first_label = str(row[0]).strip().lower() if n_cols >= 1 else ""
            second_label = str(row[1]).strip().lower() if n_cols >= 2 else ""
            labels = {first_label, second_label}
            if "gap" in labels or "total - sale" in labels:
                bg, fg, font = "#ffedd5", "#9a3412", get_pil_font(max(18, cell_font.size), bold=True) if hasattr(cell_font, "size") else font

        for c_idx, cell in enumerate(row):
            cw = col_widths[c_idx]
            draw.rectangle([cx, y, cx + cw, y + row_h], fill=bg, outline="#cbd5e1", width=2)
            text = str(cell)
            max_chars = max(8, int(cw / 18))
            if len(text) > max_chars:
                text = text[: max_chars - 1] + "…"
            use_font = font
            if first_col_bold and not is_header and c_idx == 0:
                use_font = get_pil_font(max(18, cell_font.size), bold=True) if hasattr(cell_font, "size") else font
            draw_centered_text(draw, (cx + 8, y + 8, cx + cw - 8, y + row_h - 8), text, use_font, fg)
            cx += cw
        y += row_h
    return y


def make_high_res_table_page(
    display_df: pd.DataFrame,
    title: str,
    subtitle: str,
    col_weights: Optional[list[float]] = None,
    row_h: int = 105,
) -> Image.Image:
    """Create one A4 landscape 300 DPI PNG page for a table."""
    width, height = A4_LANDSCAPE_PX
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin_x = 105
    y = 90
    title_font = get_pil_font(60, bold=True)
    subtitle_font = get_pil_font(30)
    header_font = get_pil_font(28, bold=True)
    cell_font = get_pil_font(27)

    draw.text((margin_x, y), title, font=title_font, fill="#0f172a")
    y += 78
    draw.text((margin_x, y), subtitle[:180], font=subtitle_font, fill="#475569")
    y += 82

    table_width = width - margin_x * 2
    draw_table_on_image(draw, display_df, margin_x, y, table_width, row_h, header_font, cell_font, col_weights)
    return img

def make_high_res_overview_page(
    province: str,
    cabin_name: str,
    cabin_type: str,
    ranking_month: str,
    yearly_kpi_df: pd.DataFrame,
    loss_by_year: Dict[int, list[float]],
) -> Image.Image:
    """Create A4 landscape overview page with KPI table."""
    width, height = A4_LANDSCAPE_PX
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin_x = 105
    y = 80
    title_font = get_pil_font(68, bold=True)
    subtitle_font = get_pil_font(34)
    note_font = get_pil_font(27)
    header_font = get_pil_font(30, bold=True)
    cell_font = get_pil_font(30)

    subtitle = f"Province: {province} | Cabin: {cabin_name} | Type: {cabin_type} | Ranking month: {ranking_month} {LATEST_YEAR}"
    draw.text((margin_x, y), "EDC Cabin Loss Printable Report", font=title_font, fill="#0f172a")
    y += 92
    draw.text((margin_x, y), subtitle[:185], font=subtitle_font, fill="#334155")
    y += 54
    draw.text((margin_x, y), "Print-ready export: A4 landscape. Each yearly summary fits on one clear page.", font=note_font, fill="#64748b")
    y += 92

    kpi_display = format_yearly_kpi_table(yearly_kpi_df)
    draw.text((margin_x, y), "Yearly KPI Comparison", font=get_pil_font(44, bold=True), fill="#0f172a")
    y += 66
    draw_table_on_image(
        draw,
        kpi_display,
        margin_x,
        y,
        width - margin_x * 2,
        96,
        header_font,
        cell_font,
        col_weights=[0.8, 1.45, 1.45, 1.45, 1.35, 1.55],
    )
    return img


def make_high_res_all_summary_tables_one_page(
    province: str,
    cabin_name: str,
    cabin_type: str,
    ranking_month: str,
    summary_by_year: Dict[int, pd.DataFrame],
    yearly_kpi_df: pd.DataFrame,
) -> Image.Image:
    """Create one A4 landscape page containing 2026, 2025, 2024 summaries + KPI.

    Order:
    1) Summary Table 2026
    2) Summary Table 2025
    3) Summary Table 2024
    4) Yearly KPI Comparison

    The font is kept readable by using a compact but clear row height.
    """
    width, height = A4_LANDSCAPE_PX
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin_x = 58
    y = 46
    title_font = get_pil_font(56, bold=True)
    subtitle_font = get_pil_font(28)
    section_font = get_pil_font(32, bold=True)
    header_font = get_pil_font(28, bold=True)
    cell_font = get_pil_font(27)

    subtitle = f"Province: {province} | Cabin: {cabin_name} | Type: {cabin_type} | Ranking month: {ranking_month} {LATEST_YEAR}"

    def _draw_center_line(y_pos: int, line_text: str, font, fill: str):
        bbox = draw.textbbox((0, 0), line_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) / 2, y_pos), line_text, font=font, fill=fill)

    draw.text((margin_x, y), "Summary Tables — 2026, 2025, 2024", font=title_font, fill="#0f172a")
    y += 68
    _draw_center_line(y, subtitle[:190], subtitle_font, "#475569")
    y += 54

    table_width = width - margin_x * 2
    summary_weights = [1.22, 0.70] + [1.24] * 12 + [1.85, 1.65]

    # Three yearly summary tables must share one A4 page with KPI.
    # Keep values readable while fitting all required tables.
    summary_row_h = 96
    def _draw_summary_table_with_year_column(year: int, display_df: pd.DataFrame, y_pos: int) -> int:
        year_col_w = 100
        table_x = margin_x
        data_x = table_x + year_col_w
        data_w = table_width - year_col_w
        y_start = y_pos

        # Draw data table first.
        y_end = draw_table_on_image(
            draw,
            display_df,
            data_x,
            y_pos,
            data_w,
            summary_row_h,
            header_font,
            cell_font,
            col_weights=summary_weights,
        )

        # Draw merged year column.
        draw.rectangle(
            [table_x, y_start, table_x + year_col_w, y_end],
            fill="#e2e8f0",
            outline="#94a3b8",
            width=3,
        )

        # Year text as normal horizontal table-cell text.
        draw_centered_text(
            draw,
            (table_x, y_start, table_x + year_col_w, y_end),
            str(year),
            get_pil_font(27, bold=True),
            "#0f172a",
        )

        return y_end



    for year in [2026, 2025, 2024]:
        if year in summary_by_year:
            y = _draw_summary_table_with_year_column(
                year,
                _summary_display_full_for_report(summary_by_year[year]),
                y,
            )
        else:
            draw.text((margin_x, y), f"No {year} data available for this cabin.", font=cell_font, fill="#64748b")
            y += summary_row_h

        y += 12

    # Add Yearly KPI table under the three summaries.
    draw.text((margin_x, y), "Yearly KPI Comparison", font=section_font, fill="#0f172a")
    y += 38
    kpi_display = format_yearly_kpi_table(yearly_kpi_df)
    draw_table_on_image(
        draw,
        kpi_display,
        margin_x,
        y,
        table_width,
        72,
        get_pil_font(24, bold=True),
        get_pil_font(23),
        col_weights=[0.85, 1.45, 1.45, 1.45, 1.35, 1.55],
    )

    _draw_center_line(
        height - 34,
        "Note: Gap = Total - Sale. Loss % = (1 - Sale / Total) × 100.",
        get_pil_font(20),
        "#64748b",
    )

    return img

def build_high_res_png_pages_zip_bytes(
    province: str,
    cabin_name: str,
    cabin_type: str,
    ranking_month: str,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_by_year: Dict[int, list[float]],
    yearly_kpi_df: pd.DataFrame,
    loss_compare_df: pd.DataFrame,
) -> bytes:
    """Build a ZIP containing separate A4 landscape 300 DPI PNG pages."""
    safe_cabin = safe_filename(cabin_name)
    subtitle = f"Province: {province} | Cabin: {cabin_name} | Type: {cabin_type} | Ranking month: {ranking_month} {LATEST_YEAR}"

    pages: list[tuple[str, Image.Image]] = []
    pages.append((
        safe_filename(f"01_{province}_Cabin_{safe_cabin}_Summary_2026_2025_2024_A4_Landscape_300DPI.png"),
        make_high_res_all_summary_tables_one_page(
            province=province,
            cabin_name=cabin_name,
            cabin_type=cabin_type,
            ranking_month=ranking_month,
            summary_by_year=summary_by_year,
            yearly_kpi_df=yearly_kpi_df,
        ),
    ))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename, page_img in pages:
            zipf.writestr(filename, image_to_png_bytes(page_img, dpi=PRINT_DPI))
    return zip_buffer.getvalue()

def _pdf_cell_text(value) -> str:
    """Safe short text for ReportLab tables."""
    if value is None:
        return "-"
    text = str(value)
    return text.replace("\n", " ").strip()


def _reportlab_table(
    display_df: pd.DataFrame,
    col_widths: Optional[list[float]] = None,
    font_size: float = 8.8,
    header_font_size: float = 9.2,
    numeric_start_col: int = 1,
):
    """Create a readable ReportLab table for A4 landscape pages.

    The old PDF tried to fit the full 17-column yearly summary table on one
    page, forcing very small text. This helper is used with split tables so
    the font can stay readable on A4.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    clean_df = display_df.copy().astype(str)
    data = [clean_df.columns.astype(str).tolist()] + clean_df.values.tolist()
    data = [[_pdf_cell_text(cell) for cell in row] for row in data]

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    style_items = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), header_font_size),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("ALIGN", (numeric_start_col, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.20, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    # Light semantic row colors. Only Gap is highlighted.
    # Sale, Total, and Loss % stay plain.
    for row_idx, row in enumerate(data[1:], start=1):
        first = str(row[0]).strip().lower() if len(row) > 0 else ""
        second = str(row[1]).strip().lower() if len(row) > 1 else ""
        labels = {first, second}

        if "gap" in labels or "total - sale" in labels:
            bg, fg = "#ffedd5", "#9a3412"
            style_items.append(("FONTNAME", (0, row_idx), (-1, row_idx), "Helvetica-Bold"))
        elif first.startswith("weighted") or first.startswith("average"):
            bg, fg = "#f8fafc", "#0f172a"
            style_items.append(("FONTNAME", (0, row_idx), (-1, row_idx), "Helvetica-Bold"))
        else:
            bg, fg = None, None

        if bg:
            style_items.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor(bg)))
        if fg:
            style_items.append(("TEXTCOLOR", (0, row_idx), (-1, row_idx), colors.HexColor(fg)))

    table.setStyle(TableStyle(style_items))
    return table


def _summary_display_full_for_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compact full-year summary table for one A4 landscape page.

    Removes the No column and shortens labels so all months remain readable
    on a single landscape page.
    """
    display = format_summary_for_display(summary_df).copy()
    display = display.drop(columns=["No"], errors="ignore")
    display = display.rename(columns={"Description": "Metric", "Accumulate": "Acc", "Average": "Avg"})
    display["Metric"] = display["Metric"].replace({
        "total - sale": "Gap",
        "losses": "Loss %",
    })
    return display[["Metric", "Unit", *MONTHS, "Acc", "Avg"]]


def _summary_display_halves_for_pdf(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible helper; the report now uses full landscape pages."""
    display = format_summary_for_display(summary_df)
    first_cols = ["No", "Description", "Unit", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    second_cols = ["No", "Description", "Unit", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Accumulate", "Average"]
    return display[first_cols].copy(), display[second_cols].copy()

def _pdf_footer(canvas, doc):
    from reportlab.lib.colors import HexColor
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 18, "EDC Cabin Loss Dashboard - A4 landscape selected cabin report")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 18, f"Page {doc.page}")
    canvas.restoreState()




def _reportlab_summary_table_with_rotated_year(
    summary_df: pd.DataFrame,
    year: int,
    col_widths: list[float],
    font_size: float = 8.0,
    header_font_size: float = 8.3,
    unit_font_size: float = 6.6,
):
    """Create full-year summary table with a merged rotated Year column.

    The separate title 'Summary Table (year) - Full Year' is removed.
    Instead, the year appears as a vertical merged cell at the far-left side
    of each table.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Flowable, Table, TableStyle

    class RotatedYearText(Flowable):
        def __init__(self, text: str, font_name: str = "Helvetica-Bold", font_size: float = 11):
            super().__init__()
            self.text = str(text)
            self.font_name = font_name
            self.font_size = font_size
            self.width = 22
            self.height = 86

        def wrap(self, availWidth, availHeight):
            self.width = min(availWidth, 24)
            self.height = min(availHeight, 95)
            return self.width, self.height

        def draw(self):
            c = self.canv
            c.saveState()
            c.setFillColor(colors.HexColor("#0f172a"))
            c.setFont(self.font_name, self.font_size)
            # Rotate upward and center inside the merged year column.
            c.translate(self.width / 2 + 3, self.height / 2 - 2)
            c.rotate(90)
            text_width = c.stringWidth(self.text, self.font_name, self.font_size)
            c.drawString(-text_width / 2, -self.font_size / 3, self.text)
            c.restoreState()

    display = _summary_display_full_for_report(summary_df).copy().astype(str)
    headers = display.columns.astype(str).tolist()
    rows = display.values.tolist()

    data = [[RotatedYearText(str(year)), *headers]]
    for row in rows:
        data.append(["", *[_pdf_cell_text(cell) for cell in row]])

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")

    style_items = [
        ("SPAN", (0, 0), (0, -1)),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (0, -1), 0.25, colors.black),
        ("VALIGN", (0, 0), (0, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),

        ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (1, 0), (-1, 0), colors.white),
        ("FONTNAME", (1, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (1, 0), (-1, 0), header_font_size),
        ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (1, 1), (-1, -1), font_size),
        # PDF UNIT COLUMN VALUE FONT SIZE - edit unit_font_size in the function call below.
        # Column index 2 is Unit; rows 1 to end are the kWh/% body values.
        ("FONTSIZE", (2, 1), (2, -1), unit_font_size),
        ("LEADING", (1, 0), (-1, -1), font_size + 2.4),
        ("ALIGN", (1, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("ALIGN", (2, 1), (2, -1), "CENTER"),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (1, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.20, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]

    # Row colours. Only the Gap row is highlighted; Sale, Total, and Loss % stay plain.
    for row_idx, row in enumerate(data[1:], start=1):
        metric = str(row[1]).strip().lower() if len(row) > 1 else ""
        if metric == "gap":
            bg, fg = "#ffedd5", "#9a3412"
            style_items.append(("FONTNAME", (1, row_idx), (-1, row_idx), "Helvetica-Bold"))
        else:
            bg, fg = None, None

        if bg:
            style_items.append(("BACKGROUND", (1, row_idx), (-1, row_idx), colors.HexColor(bg)))
        if fg:
            style_items.append(("TEXTCOLOR", (1, row_idx), (-1, row_idx), colors.HexColor(fg)))

    table.setStyle(TableStyle(style_items))
    return table


def get_pdf_table_title_khmer_font(font_size: int):
    """Load Khmer font for PDF table titles only."""
    font_path = PDF_TABLE_TITLE_KHMER_FONT_PATH
    if font_path.exists():
        try:
            return ImageFont.truetype(
                str(font_path),
                font_size,
                layout_engine=ImageFont.Layout.RAQM,
            )
        except Exception:
            return ImageFont.truetype(str(font_path), font_size)

    # Fallback if Khmer font file is missing. PDF still builds.
    return get_pil_font(font_size, bold=True)


def make_pdf_table_title_image_flowable(
    text: str,
    font_size: int,
    max_width_pt: float,
    align: str = "center",
):
    """Render Khmer table title as high-resolution image for ReportLab.

    This is applied ONLY to report table/title headings. It keeps the existing
    PDF table layout, cell fonts, row colors, and values unchanged.
    """
    from reportlab.platypus import Image as RLImage

    font = get_pdf_table_title_khmer_font(font_size)
    padding_x = 28
    padding_y = 12

    probe = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    img_w = text_w + padding_x * 2
    img_h = text_h + padding_y * 2
    img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    if align == "right":
        x = img_w - padding_x - text_w
    elif align == "left":
        x = padding_x
    else:
        x = (img_w - text_w) / 2

    draw.text((x, padding_y - bbox[1]), text, font=font, fill="#0f172a")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, dpi=(300, 300))
    buf.seek(0)

    natural_w_pt = img_w * 72 / 300
    natural_h_pt = img_h * 72 / 300
    draw_w = min(max_width_pt, natural_w_pt)
    draw_h = natural_h_pt * (draw_w / natural_w_pt)

    flowable = RLImage(buf, width=draw_w, height=draw_h)
    if align == "right":
        flowable.hAlign = "RIGHT"
    elif align == "left":
        flowable.hAlign = "LEFT"
    else:
        flowable.hAlign = "CENTER"
    return flowable



def _fpdf_text_safe(value) -> str:
    """Safe PDF text for fpdf2."""
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _fpdf_latin_safe(value) -> str:
    """Keep non-title PDF text in the original/simple Helvetica-style body font."""
    text = _fpdf_text_safe(value)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _fpdf_title_font_path() -> Path:
    """Return Khmer font path used only for the two PDF titles."""
    preferred = [
        PDF_TABLE_TITLE_KHMER_FONT_PATH,
        Path("fonts/KhmerOS_siemreap.ttf"),
        Path("fonts/NotoSansKhmer-Regular.ttf"),
        Path("fonts/NotoKhmer.ttf"),
    ]
    for path in preferred:
        if path.exists():
            return path
    return preferred[0]


def _fpdf_register_khmer_title_font(pdf) -> str:
    """Register Khmer font for the two PDF titles only."""
    font_path = _fpdf_title_font_path()
    if not font_path.exists():
        raise RuntimeError(
            "Khmer title font file not found. Please add fonts/KhmerOS_siemreap.ttf "
            "or fonts/NotoSansKhmer-Regular.ttf to your GitHub repo."
        )
    pdf.add_font("khmer_title", "", str(font_path))
    return "khmer_title"


def _fpdf_enable_khmer_shaping(pdf):
    """Enable HarfBuzz shaping so Khmer titles appear correctly."""
    try:
        pdf.set_text_shaping(
            use_shaping_engine=True,
            script="khmr",
            language="khm",
        )
    except TypeError:
        pdf.set_text_shaping(True)


def _fpdf_cell_text_fit(text: str, max_chars: int) -> str:
    text = _fpdf_latin_safe(text)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _fpdf_draw_text_cell(
    pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    font_family: str,
    font_size: float,
    bold: bool = False,
    align: str = "C",
    fill_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] = (17, 24, 39),
    border_color: tuple[int, int, int] = (0, 0, 0),
    line_width: float = 0.10,
):
    """Draw one fpdf2 fixed-position table cell using the original/simple body font."""
    pdf.set_xy(x, y)
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(line_width)
    if fill_color is not None:
        pdf.set_fill_color(*fill_color)
        fill = True
    else:
        pdf.set_fill_color(255, 255, 255)
        fill = False

    pdf.rect(x, y, w, h, style="DF" if fill else "D")
    pdf.set_text_color(*text_color)
    pdf.set_font(font_family, style="B" if bold else "", size=font_size)

    text_h = font_size * 0.3528
    ty = y + max(0.8, (h - text_h) / 2 - 0.2)
    pdf.set_xy(x + 0.6, ty)
    pdf.cell(w - 1.2, text_h + 1.0, _fpdf_latin_safe(text), border=0, align=align)


def _fpdf_summary_display_for_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact full-year summary table for the PDF."""
    return _summary_display_full_for_report(summary_df)


def _fpdf_draw_summary_table_with_year(
    pdf,
    display_df: pd.DataFrame,
    year: int,
    x: float,
    y: float,
    table_w: float,
    body_font_family: str,
    body_font_size: float = 7.2,
    header_font_size: float = 7.4,
    unit_font_size: float = 5.8,
) -> float:
    """Draw one yearly summary table.

    Only the two main PDF titles use Khmer font. This table uses the
    original/simple Helvetica-style body font for cells and values.
    """
    # Columns: Year | Metric | Unit | Jan-Dec | Acc | Avg
    weights = [0.50, 0.55, 0.32] + [0.72] * 12 + [0.72, 0.72]
    total_weight = sum(weights)
    widths = [table_w * weight / total_weight for weight in weights]

    row_h = 6.6
    header_h = 6.8
    rows = [display_df.columns.astype(str).tolist()] + display_df.astype(str).values.tolist()
    table_h = header_h + row_h * (len(rows) - 1)

    year_w = widths[0]
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.10)
    pdf.set_fill_color(226, 232, 240)
    pdf.rect(x, y, year_w, table_h, style="DF")

    # Year text as normal horizontal table-cell text.
    pdf.set_font(body_font_family, style="B", size=body_font_size)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(x, y + table_h / 2 - 2.5)
    pdf.cell(year_w, 5, str(year), border=0, align="C")

    data_x = x + year_w
    col_x = data_x

    headers = rows[0]
    for col_idx, header in enumerate(headers):
        w = widths[col_idx + 1]
        _fpdf_draw_text_cell(
            pdf, col_x, y, w, header_h,
            _fpdf_cell_text_fit(header, 12),
            body_font_family,
            header_font_size,
            bold=True,
            align="C",
            fill_color=(15, 23, 42),
            text_color=(255, 255, 255),
            border_color=(0, 0, 0),
            line_width=0.10,
        )
        col_x += w

    for r_idx, row in enumerate(rows[1:], start=1):
        metric = str(row[0]).strip().lower() if len(row) else ""
        y_row = y + header_h + (r_idx - 1) * row_h

        is_gap = metric in {"gap", "total - sale"}
        fill_color = (255, 237, 213) if is_gap else None
        text_color = (154, 52, 18) if is_gap else (17, 24, 39)
        row_bold = is_gap

        col_x = data_x
        for c_idx, cell in enumerate(row):
            w = widths[c_idx + 1]
            align = "L" if c_idx == 0 else ("C" if c_idx == 1 else "R")
            f_size = unit_font_size if c_idx == 1 else body_font_size
            max_chars = 9 if c_idx >= 2 else 14

            _fpdf_draw_text_cell(
                pdf,
                col_x,
                y_row,
                w,
                row_h,
                _fpdf_cell_text_fit(cell, max_chars),
                body_font_family,
                f_size,
                bold=row_bold or (c_idx == 0),
                align=align,
                fill_color=fill_color,
                text_color=text_color,
                border_color=(0, 0, 0),
                line_width=0.10,
            )
            col_x += w

    return y + table_h


def _fpdf_draw_kpi_table(
    pdf,
    kpi_df: pd.DataFrame,
    x: float,
    y: float,
    table_w: float,
    body_font_family: str,
) -> float:
    display_df = format_yearly_kpi_table(kpi_df)
    if display_df.empty:
        return y

    headers = display_df.columns.astype(str).tolist()
    rows = [headers] + display_df.astype(str).values.tolist()

    weights = [0.8, 1.45, 1.45, 1.45, 1.35, 1.55]
    total_weight = sum(weights)
    widths = [table_w * weight / total_weight for weight in weights]

    header_h = 7.2
    row_h = 7.0

    for r_idx, row in enumerate(rows):
        col_x = x
        is_header = r_idx == 0
        is_total_row = (not is_header) and str(row[0]).lower().startswith(("weighted", "average"))
        y_row = y + (header_h if r_idx > 0 else 0) + max(0, r_idx - 1) * row_h

        for c_idx, cell in enumerate(row):
            fill_color = (15, 23, 42) if is_header else ((248, 250, 252) if is_total_row else None)
            text_color = (255, 255, 255) if is_header else (17, 24, 39)
            f_size = 7.2 if is_header else 7.0
            align = "C" if c_idx == 0 else "R"

            _fpdf_draw_text_cell(
                pdf,
                col_x,
                y_row,
                widths[c_idx],
                header_h if is_header else row_h,
                _fpdf_cell_text_fit(cell, 18),
                body_font_family,
                f_size,
                bold=is_header or is_total_row,
                align=align,
                fill_color=fill_color,
                text_color=text_color,
                border_color=(0, 0, 0),
                line_width=0.10,
            )
            col_x += widths[c_idx]

    return y + header_h + row_h * (len(rows) - 1)




def make_printable_selected_report_pdf_bytes(
    province: str,
    cabin_name: str,
    region_name: str,
    cabin_type: str,
    ranking_month: str,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_by_year: Dict[int, list[float]],
    yearly_kpi_df: pd.DataFrame,
    loss_compare_df: pd.DataFrame,
) -> bytes:
    """Build an A4 landscape PDF.

    Khmer font/shaping is used ONLY for:
    - PDF_SUMMARY_TABLE_TITLE_TEXT
    - PDF_KPI_TABLE_TITLE_TEXT

    Everything else uses the original/simple Helvetica-style PDF body font.
    """
    try:
        from fpdf import FPDF
    except Exception as exc:
        raise RuntimeError(
            "PDF export requires fpdf2. Add 'fpdf2', 'uharfbuzz', and 'fonttools' to requirements.txt."
        ) from exc

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False, margin=7)
    pdf.set_margins(8, 7, 8)
    pdf.add_page()
    pdf.c_margin = 0

    khmer_title_font = _fpdf_register_khmer_title_font(pdf)
    _fpdf_enable_khmer_shaping(pdf)

    body_font = "helvetica"

    page_w = pdf.w
    page_h = pdf.h
    margin_x = 8
    usable_w = page_w - margin_x * 2

    pdf.set_font(body_font, style="B", size=10.5)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(page_w - 78, 6.2)
    pdf.multi_cell(
        70,
        5.2,
        f"Province: {_fpdf_latin_safe(province)}\nCabin: {_fpdf_latin_safe(cabin_name)}",
        border=0,
        align="R",
    )

    # Khmer title 1: fpdf2 + uharfbuzz shaping.
    pdf.set_font(khmer_title_font, style="", size=16)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(margin_x, 10.2)
    pdf.cell(usable_w, 8.0, PDF_SUMMARY_TABLE_TITLE_TEXT, border=0, align="C")

    subtitle = (
        f"Region: {_fpdf_latin_safe(region_name)} | "
        f"Type: {_fpdf_latin_safe(cabin_type)} | "
        f"Ranking month: {_fpdf_latin_safe(ranking_month)} {LATEST_YEAR}"
    )
    pdf.set_font(body_font, style="", size=7.4)
    pdf.set_text_color(71, 85, 105)
    pdf.set_xy(margin_x, 19.8)
    pdf.cell(usable_w, 5.0, subtitle, border=0, align="C")

    y = 26.0

    for year in [2026, 2025, 2024]:
        if year in summary_by_year:
            display_df = _fpdf_summary_display_for_report(summary_by_year[year])
            y = _fpdf_draw_summary_table_with_year(
                pdf,
                display_df,
                year,
                margin_x,
                y,
                usable_w,
                body_font,
                body_font_size=7.2,
                header_font_size=7.4,
                unit_font_size=5.8,
            )
        else:
            pdf.set_font(body_font, style="", size=8)
            pdf.set_text_color(100, 116, 139)
            pdf.set_xy(margin_x, y)
            pdf.cell(usable_w, 7, _fpdf_latin_safe(f"No {year} data available for this cabin."), border=0, align="L")
            y += 7
        y += 2.5

    # Khmer title 2: fpdf2 + uharfbuzz shaping.
    pdf.set_font(khmer_title_font, style="", size=12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(margin_x, y + 0.5)
    pdf.cell(usable_w, 6, PDF_KPI_TABLE_TITLE_TEXT, border=0, align="C")
    y += 7.3

    _fpdf_draw_kpi_table(
        pdf,
        yearly_kpi_df,
        margin_x + 36,
        y,
        usable_w - 72,
        body_font,
    )

    pdf.set_font(body_font, style="", size=6.8)
    pdf.set_text_color(100, 116, 139)
    pdf.set_xy(margin_x, page_h - 8.2)
    pdf.cell(
        usable_w,
        4.0,
        "Note: Gap = Total - Sale. Loss % is calculated as (1 - Sale / Total) x 100.",
        border=0,
        align="C",
    )

    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        return bytes(output)
    if isinstance(output, bytes):
        return output
    return str(output).encode("latin1")


def build_selected_pdf_report_package_zip_bytes(
    province: str,
    cabin_name: str,
    ranking_month: str,
    report_pdf_bytes: bytes,
    visible_ranking_df: pd.DataFrame,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_compare_df: pd.DataFrame,
    yearly_kpi_df: pd.DataFrame,
) -> bytes:
    """ZIP package with the sharp PDF report and supporting CSV files."""
    zip_buffer = io.BytesIO()
    safe_cabin = safe_filename(cabin_name)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_{ranking_month}_{LATEST_YEAR}_Printable_Report.pdf"),
            report_pdf_bytes,
        )
        zipf.writestr(
            safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_Visible_Ranking.csv"),
            build_ranking_csv_bytes(visible_ranking_df),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Summary_2024_2025_2026.csv"),
            build_all_year_summary_csv(summary_by_year),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Loss_Comparison.csv"),
            loss_compare_df.to_csv(index=False).encode("utf-8-sig"),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Yearly_KPI.csv"),
            yearly_kpi_df.to_csv(index=False).encode("utf-8-sig"),
        )
    return zip_buffer.getvalue()


def is_reportlab_available() -> bool:
    """Return True only when fpdf2 + uharfbuzz PDF support is installed."""
    try:
        import fpdf  # noqa: F401
        import uharfbuzz  # noqa: F401
        return True
    except Exception:
        return False


def build_selected_print_report_package_zip_bytes(
    province: str,
    cabin_name: str,
    ranking_month: str,
    report_pdf_bytes: Optional[bytes],
    png_pages_zip_bytes: bytes,
    visible_ranking_df: pd.DataFrame,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_compare_df: pd.DataFrame,
    yearly_kpi_df: pd.DataFrame,
) -> bytes:
    """ZIP package that never depends on reportlab.

    If PDF bytes exist, include the PDF. Always include the A4 300 DPI PNG
    pages ZIP and CSV support files so the export still works when reportlab
    is missing from Streamlit Cloud.
    """
    zip_buffer = io.BytesIO()
    safe_cabin = safe_filename(cabin_name)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        if report_pdf_bytes:
            zipf.writestr(
                safe_filename(f"{province}_Cabin_{safe_cabin}_{ranking_month}_{LATEST_YEAR}_Printable_Report.pdf"),
                report_pdf_bytes,
            )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_{ranking_month}_{LATEST_YEAR}_A4_300DPI_png_pages.zip"),
            png_pages_zip_bytes,
        )
        zipf.writestr(
            safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_Visible_Ranking.csv"),
            build_ranking_csv_bytes(visible_ranking_df),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Summary_2024_2025_2026.csv"),
            build_all_year_summary_csv(summary_by_year),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Loss_Comparison.csv"),
            loss_compare_df.to_csv(index=False).encode("utf-8-sig"),
        )
        zipf.writestr(
            safe_filename(f"{province}_Cabin_{safe_cabin}_Yearly_KPI.csv"),
            yearly_kpi_df.to_csv(index=False).encode("utf-8-sig"),
        )
    return zip_buffer.getvalue()


def build_selected_cabin_csv_bytes(
    province: str,
    cabin_name: str,
    ranking_month: str,
    summary_by_year: Dict[int, pd.DataFrame],
    loss_compare_df: pd.DataFrame,
    yearly_kpi_df: pd.DataFrame,
) -> bytes:
    """Build one clean CSV for the selected cabin only.

    The previous implementation used fixed part indexes to control headers,
    which could suppress dataframe headers. This version writes each section
    explicitly so the exported CSV is easier to read in Excel.
    """
    buffer = io.StringIO()

    def _write_section(title: str, df: pd.DataFrame):
        buffer.write(title + "\n")
        if df is None or df.empty:
            buffer.write("No data available\n\n")
            return
        df.to_csv(buffer, index=False)
        buffer.write("\n")

    summary_frames = []
    for year in [2026, 2025, 2024]:
        if year in summary_by_year:
            temp = summary_by_year[year].copy()
            temp.insert(0, "Year", year)
            summary_frames.append(temp)

    if summary_frames:
        selected_summary = pd.concat(summary_frames, ignore_index=True)
        selected_summary.insert(0, "Province", province)
        selected_summary.insert(1, "Cabin", cabin_name)
        selected_summary.insert(2, "Ranking Month", f"{ranking_month} {LATEST_YEAR}")
    else:
        selected_summary = pd.DataFrame()

    _write_section("SELECTED CABIN YEARLY SUMMARY", selected_summary)
    _write_section("MONTHLY LOSS COMPARISON", loss_compare_df)
    _write_section("YEARLY KPI COMPARISON", yearly_kpi_df)

    return buffer.getvalue().encode("utf-8-sig")


def build_batch_pdf_reports_zip_bytes(
    province: str,
    ranking_month: str,
    ranking_df: pd.DataFrame,
    year_sources: Dict[int, Dict[str, CsvSource]],
    data_2026: Dict[str, pd.DataFrame],
) -> bytes:
    """Build a lightweight ZIP containing one sharp PDF per selected ranked cabin.

    This intentionally avoids PNG pages inside the ZIP so the file stays smaller.
    The PDFs remain clear because they are generated as PDF tables, not screenshots.
    """
    zip_buffer = io.BytesIO()
    manifest_rows = []
    error_rows = []

    meta_2026_indexed_local = data_2026["cabin_meta_indexed"]

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for _, rank_row in ranking_df.iterrows():
            try:
                cabin_key = rank_row["__cabin_key"]
                selected_row_2026 = get_row_from_meta(meta_2026_indexed_local, cabin_key)
                if selected_row_2026 is None:
                    error_rows.append({
                        "Rank": rank_row.get("Rank", ""),
                        "Cabin": rank_row.get("display_name", ""),
                        "Error": "Cabin not found in 2026 metadata",
                    })
                    continue

                cabin_name = str(selected_row_2026["display_name"])
                region_name = normalize_text(selected_row_2026.get("region", ""))
                if region_name == "":
                    region_name = normalize_text(selected_row_2026.get("Region", ""))

                summary_by_year_i, loss_by_year_i, _, status_by_year_i = get_summary_and_loss_for_cabin(
                    year_sources=year_sources,
                    province=province,
                    selected_cabin_key=cabin_key,
                    data_2026=data_2026,
                    selected_row_2026=selected_row_2026,
                )

                yearly_kpi_df_i = build_yearly_kpi_table(summary_by_year_i)
                loss_compare_df_i = build_loss_comparison_table(summary_by_year_i, loss_by_year_i)

                report_pdf_i = make_printable_selected_report_pdf_bytes(
                    province=province,
                    cabin_name=cabin_name,
                    region_name=region_name,
                    cabin_type=str(selected_row_2026["type"]),
                    ranking_month=ranking_month,
                    summary_by_year=summary_by_year_i,
                    loss_by_year=loss_by_year_i,
                    yearly_kpi_df=yearly_kpi_df_i,
                    loss_compare_df=loss_compare_df_i,
                )

                rank_no = int(rank_row.get("Rank", 0))
                safe_cabin = safe_filename(cabin_name)
                pdf_name = safe_filename(
                    f"Rank_{rank_no:03d}_{province}_Cabin_{safe_cabin}_{ranking_month}_{LATEST_YEAR}.pdf"
                )
                zipf.writestr(pdf_name, report_pdf_i)

                manifest_rows.append({
                    "Rank": rank_no,
                    "Province": province,
                    "Cabin": cabin_name,
                    "Region": region_name,
                    "Type": str(selected_row_2026["type"]),
                    "PDF File": pdf_name,
                    "2026 Status": status_by_year_i.get(2026, "-"),
                    "2025 Status": status_by_year_i.get(2025, "-"),
                    "2024 Status": status_by_year_i.get(2024, "-"),
                })

            except Exception as exc:
                error_rows.append({
                    "Rank": rank_row.get("Rank", ""),
                    "Cabin": rank_row.get("display_name", ""),
                    "Error": str(exc),
                })

        if manifest_rows:
            zipf.writestr(
                safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_batch_manifest.csv"),
                pd.DataFrame(manifest_rows).to_csv(index=False).encode("utf-8-sig"),
            )

        if error_rows:
            zipf.writestr(
                safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_batch_errors.csv"),
                pd.DataFrame(error_rows).to_csv(index=False).encode("utf-8-sig"),
            )

    return zip_buffer.getvalue()



# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <div class="edc-hero">
        <h1>⚡ EDC Cabin Loss Dashboard</h1>
        <p>Analyze single-cabin ranking, compare 2024–2026 loss performance, and export print-ready reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

static_sources = build_static_sources()

with st.sidebar:
    st.header("Data input")
    uploaded_2026_files = st.file_uploader(
        "Upload 2026 province CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload all 15 province CSV files. Best filenames: BMC.csv, BTB.csv, KP.csv, etc.",
    )

uploaded_2026_sources, rejected_uploads, duplicate_uploads = build_uploaded_2026_sources(uploaded_2026_files)
year_sources: Dict[int, Dict[str, CsvSource]] = {
    2024: static_sources.get(2024, {}),
    2025: static_sources.get(2025, {}),
    2026: uploaded_2026_sources,
}

with st.sidebar:
    st.subheader("Upload status")
    uploaded_count = len(uploaded_2026_sources)
    missing_2026 = [p for p in PROVINCES if p not in uploaded_2026_sources]
    missing_2024 = [p for p in PROVINCES if p not in year_sources[2024]]
    missing_2025 = [p for p in PROVINCES if p not in year_sources[2025]]

    st.write(f"2026 uploaded provinces: **{uploaded_count}/15**")
    if missing_2026:
        st.warning("Missing 2026: " + ", ".join(missing_2026))
    else:
        st.success("All 15 province CSVs uploaded for 2026.")

    if rejected_uploads:
        st.error("Rejected filenames: " + ", ".join(rejected_uploads))
    if duplicate_uploads:
        st.warning("Duplicate province uploads detected; the last file was used: " + ", ".join(duplicate_uploads))

    with st.expander("Historical repository data check"):
        st.write(f"2024 files found: **{15 - len(missing_2024)}/15**")
        if missing_2024:
            st.write("Missing 2024: " + ", ".join(missing_2024))
        st.write(f"2025 files found: **{15 - len(missing_2025)}/15**")
        if missing_2025:
            st.write("Missing 2025: " + ", ".join(missing_2025))

    st.header("Analysis controls")
    available_for_analysis = [p for p in PROVINCES if p in uploaded_2026_sources]
    if not available_for_analysis:
        available_for_analysis = PROVINCES

    province = st.selectbox("Province", available_for_analysis, index=0)
    ranking_month = st.selectbox("Ranking month", MONTHS, index=MONTHS.index("Dec") if "Dec" in MONTHS else 0)
    top_n_choice = st.selectbox("Ranking rows shown", [10, 20, 50, "All"], index=2)
    cabin_type_filter = st.selectbox("Cabin type filter", ["All", "Single", "Multiple", "Unknown"], index=0)

source_2026 = get_year_source(year_sources, 2026, province)
if source_2026 is None:
    st.info("Upload the 2026 CSV file for the selected province to start analysis.")
    st.stop()

try:
    data_2026 = prepare_source(source_2026)
except Exception as exc:
    st.error(f"Could not read 2026 file for {province}: {exc}")
    st.stop()

cabin_meta_2026 = data_2026["cabin_meta"]
meta_2026_indexed = data_2026["cabin_meta_indexed"]
counts = data_2026["counts"].iloc[0].to_dict()

# ---------------------------------------------------------
# Data quality and overview
# ---------------------------------------------------------
st.subheader(f"Cabin Type Counts — {province} 2026")
metric_cols = st.columns(4)
metric_cols[0].metric("All Cabins", int(counts["All"]))
metric_cols[1].metric("Single", int(counts["Single"]))
metric_cols[2].metric("Multiple", int(counts["Multiple"]))
metric_cols[3].metric("Unknown", int(counts["Unknown"]))

unknown_share = 0 if counts["All"] == 0 else counts["Unknown"] / counts["All"] * 100
if unknown_share > 5:
    st.warning(
        f"Data quality warning: Unknown cabin type is {unknown_share:.1f}% of cabins. "
        "Check blank or invalid Consumer values before using this report for billing decisions."
    )

st.divider()

# ---------------------------------------------------------
# Ranking
# ---------------------------------------------------------
st.subheader(f"Single Cabin Ranking by Total - Sale — {ranking_month} {LATEST_YEAR}")
full_ranking_df = build_single_cabin_gap_ranking(cabin_meta_2026, ranking_month)

if top_n_choice == "All":
    visible_ranking_df = full_ranking_df.copy()
else:
    visible_ranking_df = full_ranking_df.head(int(top_n_choice)).copy()

if visible_ranking_df.empty:
    st.warning("No Single cabin ranking data found for the selected province/month.")
else:
    st.markdown("#### Ranking overview")
    rank_m1, rank_m2, rank_m3, rank_m4 = st.columns(4)
    rank_m1.metric("Visible cabins", len(visible_ranking_df))
    rank_m2.metric("Visible total gap", f"{visible_ranking_df['rank_gap'].sum():,.0f} kWh")
    rank_m3.metric("Highest gap", f"{visible_ranking_df['rank_gap'].max():,.0f} kWh")
    rank_m4.metric("Highest loss", f"{visible_ranking_df['rank_loss_pct'].max():.2f}%")

    with st.expander("View ranking table", expanded=True):
        st.dataframe(
            make_ranking_display_df(visible_ranking_df),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
    st.download_button(
        "Download visible ranking CSV",
        data=build_ranking_csv_bytes(visible_ranking_df),
        file_name=safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_visible_ranking.csv"),
        mime="text/csv",
    )

    with st.expander("Batch print-ready PDF reports", expanded=False):
        st.caption(
            "Small clear version: this builds a ZIP of PDF reports only. "
            "No PNG pages are included, so the ZIP stays smaller and the PDF tables remain sharp."
        )

        batch_scope = st.radio(
            "Cabins to include",
            ["Current visible ranking", "Top 20", "Top 50"],
            horizontal=True,
            key=f"batch_pdf_scope_{province}_{ranking_month}_{top_n_choice}",
        )

        if batch_scope == "Current visible ranking":
            batch_df = visible_ranking_df.copy()
        elif batch_scope == "Top 20":
            batch_df = full_ranking_df.head(20).copy()
        else:
            batch_df = full_ranking_df.head(50).copy()

        st.write(f"PDFs to build: **{len(batch_df)}**")

        batch_cache_key = (
            f"{province}|{ranking_month}|{batch_scope}|{len(batch_df)}|"
            "print_ready_batch_v35_debug_convenience"
        )

        for state_key in [
            "batch_pdf_cache_key",
            "batch_pdf_zip_bytes",
            "batch_pdf_error",
        ]:
            if state_key not in st.session_state:
                st.session_state[state_key] = None

        batch_pdf_supported = is_reportlab_available()

        if not batch_pdf_supported:
            st.warning(
                "Batch PDF export needs `fpdf2`, `uharfbuzz`, and `fonttools` in requirements.txt."
            )

        if st.button(
            "Build batch PDF ZIP",
            type="secondary",
            disabled=(not batch_pdf_supported or batch_df.empty),
            key=f"build_batch_pdf_zip_{province}_{ranking_month}_{top_n_choice}",
            use_container_width=True,
        ):
            st.session_state["batch_pdf_cache_key"] = None
            st.session_state["batch_pdf_zip_bytes"] = None
            st.session_state["batch_pdf_error"] = None

            try:
                with st.spinner(f"Building {len(batch_df)} PDF reports into one ZIP..."):
                    st.session_state["batch_pdf_zip_bytes"] = build_batch_pdf_reports_zip_bytes(
                        province=province,
                        ranking_month=ranking_month,
                        ranking_df=batch_df,
                        year_sources=year_sources,
                        data_2026=data_2026,
                    )
                    st.session_state["batch_pdf_cache_key"] = batch_cache_key
            except Exception as exc:
                st.session_state["batch_pdf_error"] = f"Could not build batch PDF ZIP: {exc}"

        if st.session_state.get("batch_pdf_error"):
            st.error(st.session_state["batch_pdf_error"])

        if (
            st.session_state.get("batch_pdf_cache_key") == batch_cache_key
            and st.session_state.get("batch_pdf_zip_bytes")
        ):
            st.download_button(
                "Download batch PDF ZIP",
                data=st.session_state["batch_pdf_zip_bytes"],
                file_name=safe_filename(
                    f"{province}_{ranking_month}_{LATEST_YEAR}_{batch_scope}_PDF_reports.zip"
                ),
                mime="application/zip",
                use_container_width=True,
                key=f"download_batch_pdf_zip_{province}_{ranking_month}_{top_n_choice}",
            )

st.divider()

# ---------------------------------------------------------
# Cabin selector
# ---------------------------------------------------------
if cabin_type_filter == "All":
    cabin_options_df = cabin_meta_2026.copy()
else:
    cabin_options_df = cabin_meta_2026[cabin_meta_2026["type"] == cabin_type_filter].copy()

if cabin_options_df.empty:
    st.warning("No cabin found for the selected cabin type filter.")
    st.stop()

# Default cabin comes from the highest visible ranking row when available.
ranked_default_key = visible_ranking_df.iloc[0]["__cabin_key"] if not visible_ranking_df.empty else cabin_options_df.iloc[0]["__cabin_key"]
default_row = get_row_from_meta(meta_2026_indexed, ranked_default_key)
default_cabin_text = str(default_row["display_name"]) if default_row is not None else str(cabin_options_df.iloc[0]["display_name"])

# Stable session key lets the buttons and spinner update the typed search box.
cabin_search_key = f"cabin_search_{province}_{cabin_type_filter}"
pending_cabin_search_key = f"pending_cabin_search_{province}_{cabin_type_filter}"

# Streamlit rule: once a widget is created in a run, its session_state value
# cannot be changed later in that same run. Navigation buttons therefore write
# to a separate pending key, then rerun. On the next run, before the text_input
# widget is created, we safely apply the pending cabin value.
if pending_cabin_search_key in st.session_state:
    st.session_state[cabin_search_key] = st.session_state.pop(pending_cabin_search_key)
elif cabin_search_key not in st.session_state:
    st.session_state[cabin_search_key] = default_cabin_text

st.markdown("### Open cabin summary")
st.caption(
    "Type a cabin number/name, open the top ranked cabin, or use the stable ▲ / ▼ buttons beside the summary tables."
)

# First row: typed search + quick actions.
col_search, col_available, col_top = st.columns([2.4, 0.8, 0.9])
with col_search:
    st.text_input(
        "Cabin number/name",
        placeholder="Example: 1234 or Cabin name",
        key=cabin_search_key,
    )

with col_available:
    st.metric("Available under filter", len(cabin_options_df))

with col_top:
    if st.button(
        "Open top ranked",
        use_container_width=True,
        disabled=visible_ranking_df.empty,
        key=f"open_top_ranked_{province}_{ranking_month}_{top_n_choice}_{cabin_type_filter}",
    ):
        top_key = visible_ranking_df.iloc[0]["__cabin_key"]
        top_row = get_row_from_meta(meta_2026_indexed, top_key)
        if top_row is not None:
            st.session_state[pending_cabin_search_key] = str(top_row["display_name"])
            st.rerun()

# Final resolution used by the rest of the report.
cabin_query = st.session_state.get(cabin_search_key, "")
selected_cabin_key, cabin_message, cabin_matches_df = resolve_cabin_text_input(cabin_options_df, cabin_query)

if selected_cabin_key is None:
    st.warning(cabin_message)
    if not cabin_matches_df.empty:
        st.dataframe(cabin_matches_df, use_container_width=True, hide_index=True)
    st.stop()
else:
    st.success(cabin_message)

selected_row = get_row_from_meta(meta_2026_indexed, selected_cabin_key)
if selected_row is None:
    st.error("Selected cabin was not found in 2026 data.")
    st.stop()

resolved_name = selected_row["display_name"]
resolved_region = normalize_text(selected_row.get("region", ""))
if resolved_region == "":
    resolved_region = normalize_text(selected_row.get("Region", ""))


summary_by_year, loss_by_year, meta_by_year, status_by_year = get_summary_and_loss_for_cabin(
    year_sources=year_sources,
    province=province,
    selected_cabin_key=selected_cabin_key,
    data_2026=data_2026,
    selected_row_2026=selected_row,
)

yearly_kpi_df = build_yearly_kpi_table(summary_by_year)
loss_compare_df = build_loss_comparison_table(summary_by_year, loss_by_year)

# ---------------------------------------------------------
# Selected cabin overview
# ---------------------------------------------------------
st.subheader(f"Selected Cabin Overview — {province} | Cabin {resolved_name}")
cols = st.columns(6)
cols[0].metric("Province", province)
cols[1].metric("Region", resolved_region if resolved_region else "-")
cols[2].metric("Cabin", resolved_name)
cols[3].metric("Cabin Type", selected_row["type"])
cols[4].metric("Matched Rows", int(selected_row["rows"]))
cols[5].metric("Customers", int(selected_row["customers"]))

for year in [2025, 2024]:
    if status_by_year.get(year) != "OK":
        st.warning(f"{year}: {status_by_year.get(year)}")

overview_tab, chart_tab = st.tabs(["KPI & comparison", "Charts"])

with overview_tab:
    st.markdown("#### Yearly KPI Comparison")
    st.dataframe(format_yearly_kpi_table(yearly_kpi_df), use_container_width=True, hide_index=True)

    st.markdown("#### Monthly Loss % Comparison — 2024 / 2025 / 2026")
    st.dataframe(format_loss_comparison_table(loss_compare_df), use_container_width=True, hide_index=True)

with chart_tab:
    st.markdown("#### Loss % Trend")
    st.plotly_chart(build_multi_year_loss_chart(loss_by_year), use_container_width=True)

    st.markdown("#### Sale vs Total — 2026")
    st.plotly_chart(build_sale_total_chart(summary_by_year[2026], 2026), use_container_width=True)

st.markdown("### Summary tables & actions")

# Cabin navigation is intentionally placed beside the summary tables, not above the report.
# IMPORTANT: The browse list follows the visible ranking table only.
# Therefore, changing Ranking month or Ranking rows shown changes the cabins browsed here.
browse_df = visible_ranking_df.reset_index(drop=True).copy()
browse_keys = browse_df["__cabin_key"].tolist() if not browse_df.empty else []
rank_label_map = {}
for _, r in browse_df.iterrows():
    rank_label_map[r["__cabin_key"]] = (
        f"Rank {int(r['Rank']):03d} | Cabin {r['display_name']} | "
        f"Gap {format_number(r['rank_gap'], 0)} kWh | Loss {format_percent(r['rank_loss_pct'])}"
    )

if browse_keys and selected_cabin_key in browse_keys:
    current_pos = browse_keys.index(selected_cabin_key)
elif browse_keys:
    current_pos = 0
else:
    current_pos = -1

summary_table_col, summary_control_col = st.columns([3.7, 1.1])

with summary_control_col:
    # Keep the browse controls visually stable while moving through ranked cabins.
    # The content is grouped in a bordered panel, and the arrow buttons use callbacks.
    # The buttons use on_click callbacks instead of manual st.rerun(), so Streamlit
    # updates the selected cabin cleanly on the next run.
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(.browse-ranked-anchor) {
            position: sticky;
            top: 0.75rem;
            z-index: 20;
            background: white;
            padding-top: 0.25rem;
        }
        </style>
        <div class="browse-ranked-anchor"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Browse ranked cabins")
    st.caption(
        f"Use ▲ / ▼ to move inside the visible ranking range: "
        f"{ranking_month} {LATEST_YEAR}, rows shown = {top_n_choice}."
    )

    def _open_ranked_cabin_at_position(target_pos: int):
        if not browse_keys:
            return
        target_pos = max(0, min(len(browse_keys) - 1, int(target_pos)))
        target_key = browse_keys[target_pos]
        target_row = get_row_from_meta(meta_2026_indexed, target_key)
        if target_row is not None:
            st.session_state[pending_cabin_search_key] = str(target_row["display_name"])
            st.session_state[
                f"browse_rank_pos_{province}_{ranking_month}_{top_n_choice}_{cabin_type_filter}"
            ] = target_pos

    if not browse_keys:
        st.info("No visible ranking cabins to browse. Check the ranking month or selected province.")
    else:
        browse_pos_key = f"browse_rank_pos_{province}_{ranking_month}_{top_n_choice}_{cabin_type_filter}"

        if selected_cabin_key in browse_keys:
            current_pos = browse_keys.index(selected_cabin_key)
            st.session_state[browse_pos_key] = current_pos
        else:
            current_pos = int(st.session_state.get(browse_pos_key, 0))
            current_pos = max(0, min(len(browse_keys) - 1, current_pos))
            st.session_state[browse_pos_key] = current_pos
            st.warning(
                "The typed cabin is outside the visible ranking range. "
                "Use ▲ / ▼ below to open a ranked cabin."
            )

        current_key = browse_keys[current_pos]
        current_rank_text = f"{current_pos + 1}/{len(browse_keys)}"
        selected_rank_label = rank_label_map.get(current_key, current_key)

        st.metric("Visible rank position", current_rank_text)
        st.caption(f"Current browse target: {selected_rank_label}")

        up_col, down_col = st.columns(2)
        with up_col:
            st.button(
                "▲ Up",
                key=f"summary_rank_up_{province}_{ranking_month}_{top_n_choice}_{cabin_type_filter}",
                use_container_width=True,
                disabled=current_pos <= 0,
                on_click=_open_ranked_cabin_at_position,
                args=(current_pos - 1,),
                help="Move to the previous/higher ranked cabin.",
            )
        with down_col:
            st.button(
                "▼ Down",
                key=f"summary_rank_down_{province}_{ranking_month}_{top_n_choice}_{cabin_type_filter}",
                use_container_width=True,
                disabled=current_pos >= len(browse_keys) - 1,
                on_click=_open_ranked_cabin_at_position,
                args=(current_pos + 1,),
                help="Move to the next/lower ranked cabin.",
            )

        with st.expander("Jump to a ranked cabin", expanded=False):
            ranked_choice_key = st.selectbox(
                "Choose visible ranked cabin",
                options=browse_keys,
                index=current_pos,
                format_func=lambda key: rank_label_map.get(key, key),
                key=f"summary_visible_rank_choice_{province}_{ranking_month}_{top_n_choice}",
            )

            if st.button(
                "Open chosen ranked cabin",
                key=f"summary_open_visible_rank_choice_{province}_{ranking_month}_{top_n_choice}",
                use_container_width=True,
            ):
                jump_pos = browse_keys.index(ranked_choice_key)
                _open_ranked_cabin_at_position(jump_pos)
                st.rerun()

        st.markdown("#### Print-ready report")
        st.caption("Selected cabin only: build once, then download PDF report or CSV.")

        selected_report_key = (
            f"{province}|{ranking_month}|{selected_cabin_key}|"
            f"{','.join(map(str, sorted(summary_by_year.keys())))}|"
            "print_ready_selected_v35_debug_convenience"
        )

        for state_key in [
            "selected_report_cache_key",
            "selected_report_pdf_bytes",
            "selected_report_csv_bytes",
            "selected_report_error",
        ]:
            if state_key not in st.session_state:
                st.session_state[state_key] = None

        selected_pdf_supported = is_reportlab_available()

        if st.button(
            "Build print-ready report",
            type="primary",
            disabled=not selected_pdf_supported,
            key=f"build_selected_print_ready_{province}_{ranking_month}_{selected_cabin_key}",
            use_container_width=True,
        ):
            st.session_state["selected_report_cache_key"] = None
            st.session_state["selected_report_pdf_bytes"] = None
            st.session_state["selected_report_csv_bytes"] = None
            st.session_state["selected_report_error"] = None

            try:
                with st.spinner("Building selected cabin PDF report and CSV..."):
                    selected_pdf_bytes = make_printable_selected_report_pdf_bytes(
                        province=province,
                        cabin_name=resolved_name,
                        region_name=resolved_region,
                        cabin_type=str(selected_row["type"]),
                        ranking_month=ranking_month,
                        summary_by_year=summary_by_year,
                        loss_by_year=loss_by_year,
                        yearly_kpi_df=yearly_kpi_df,
                        loss_compare_df=loss_compare_df,
                    )
                    selected_csv_bytes = build_selected_cabin_csv_bytes(
                        province=province,
                        cabin_name=resolved_name,
                        ranking_month=ranking_month,
                        summary_by_year=summary_by_year,
                        loss_compare_df=loss_compare_df,
                        yearly_kpi_df=yearly_kpi_df,
                    )

                    st.session_state["selected_report_cache_key"] = selected_report_key
                    st.session_state["selected_report_pdf_bytes"] = selected_pdf_bytes
                    st.session_state["selected_report_csv_bytes"] = selected_csv_bytes
            except Exception as exc:
                st.session_state["selected_report_error"] = f"Could not build selected report: {exc}"

        if not selected_pdf_supported:
            st.warning(
                "PDF export needs `fpdf2`, `uharfbuzz`, and `fonttools` in requirements.txt."
            )

        if st.session_state.get("selected_report_error"):
            st.error(st.session_state["selected_report_error"])

        selected_report_ready = (
            st.session_state.get("selected_report_cache_key") == selected_report_key
            and st.session_state.get("selected_report_pdf_bytes")
            and st.session_state.get("selected_report_csv_bytes")
        )

        if selected_report_ready:
            safe_cabin_name_rank = safe_filename(resolved_name)
            st.success("Selected cabin report is ready.")
            st.download_button(
                "Download PDF report",
                data=st.session_state["selected_report_pdf_bytes"],
                file_name=safe_filename(
                    f"{province}_Cabin_{safe_cabin_name_rank}_{ranking_month}_{LATEST_YEAR}_printable_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
                key=f"download_selected_pdf_{province}_{ranking_month}_{selected_cabin_key}",
            )
            st.download_button(
                "Download selected cabin CSV",
                data=st.session_state["selected_report_csv_bytes"],
                file_name=safe_filename(
                    f"{province}_Cabin_{safe_cabin_name_rank}_{ranking_month}_{LATEST_YEAR}_selected_cabin.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key=f"download_selected_csv_{province}_{ranking_month}_{selected_cabin_key}",
            )

with summary_table_col:
    for year in [2026, 2025, 2024]:
        st.markdown(f"#### Summary Table ({year})")
        if year in summary_by_year:
            st.dataframe(format_summary_for_display(summary_by_year[year]), use_container_width=True, hide_index=True)
        else:
            st.info(status_by_year.get(year, f"No {year} data available."))

with st.expander("Matched raw rows — 2026"):
    raw_display = data_2026["df"][data_2026["df"]["__cabin_key"] == selected_cabin_key].copy()
    raw_display = raw_display.drop(columns=[c for c in raw_display.columns if c.startswith("__")], errors="ignore")
    st.dataframe(raw_display.reset_index(drop=True), use_container_width=True)

st.divider()

with st.expander("Where are the export buttons?"):
    st.write(
        "Selected cabin export is now beside the summary tables under **Print-ready report**. "
        "Batch export is now under the ranking table in **Batch print-ready PDF reports**. "
        "The batch ZIP contains PDFs only, which keeps the file smaller and avoids blurry image scaling."
    )
