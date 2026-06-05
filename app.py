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

st.set_page_config(
    page_title="EDC Cabin Loss Dashboard",
    page_icon="⚡",
    layout="wide",
)


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
    return pd.read_csv(path_str, header=None)


@st.cache_data(show_spinner=False)
def load_raw_csv_from_bytes(content: bytes, label: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), header=None)


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
            "__cabin_key", "Cabin", "display_name", "customers", "rows", "type",
            *[f"{m}_total" for m in MONTHS],
            *[f"{m}_sale" for m in MONTHS],
        ])
    else:
        cabin_meta = valid_df.groupby("__cabin_key", as_index=False).agg(
            Cabin=("Cabin", "first"),
            display_name=("__display_name", "first"),
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
    if d == "sale":
        return "#ecfdf5", "#065f46"
    if d == "total":
        return "#eff6ff", "#1d4ed8"
    if d == "total - sale":
        return "#fff7ed", "#c2410c"
    if d == "losses":
        return "#fef2f2", "#b91c1c"
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


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
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
    report_w = max([kpi_img.width, chart_img.width, loss_compare_img.width, *[img.width for img in table_images]]) + margin * 2
    report_h = header_h + kpi_img.height + gap + chart_img.height + gap + loss_compare_img.height + gap
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
    for img in [kpi_img, chart_img, loss_compare_img, *table_images]:
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
# UI
# =========================================================
st.title("⚡ EDC Cabin Loss Dashboard")
st.caption(
    "2024/2025 are loaded from the GitHub repository. Upload 2026 CSV files only when running the app."
)

static_sources = build_static_sources()

with st.sidebar:
    st.header("1) Data input")
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

    st.header("2) Analysis controls")
    available_for_analysis = [p for p in PROVINCES if p in uploaded_2026_sources]
    if not available_for_analysis:
        available_for_analysis = PROVINCES

    province = st.selectbox("Province", available_for_analysis, index=0)
    ranking_month = st.selectbox("Ranking month", MONTHS, index=0)
    top_n_choice = st.selectbox("Ranking rows shown", [10, 20, 50, "All"], index=1)
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
    st.dataframe(make_ranking_display_df(visible_ranking_df), use_container_width=True, hide_index=True)
    st.download_button(
        "Download visible ranking CSV",
        data=build_ranking_csv_bytes(visible_ranking_df),
        file_name=safe_filename(f"{province}_{ranking_month}_{LATEST_YEAR}_visible_ranking.csv"),
        mime="text/csv",
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

ranked_default_key = visible_ranking_df.iloc[0]["__cabin_key"] if not visible_ranking_df.empty else cabin_options_df.iloc[0]["__cabin_key"]
default_row = get_row_from_meta(meta_2026_indexed, ranked_default_key)
default_cabin_text = str(default_row["display_name"]) if default_row is not None else str(cabin_options_df.iloc[0]["display_name"])

st.markdown("### Open cabin summary")
st.caption("Type the cabin number/name. Exact match opens directly; partial match works only when it finds one cabin.")

col_search, col_hint = st.columns([2, 1])
with col_search:
    cabin_query = st.text_input(
        "Cabin number/name",
        value=default_cabin_text,
        placeholder="Example: 1234 or Cabin name",
        key=f"cabin_search_{province}_{cabin_type_filter}",
    )

with col_hint:
    st.metric("Available under filter", len(cabin_options_df))

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
cols = st.columns(5)
cols[0].metric("Province", province)
cols[1].metric("Cabin", resolved_name)
cols[2].metric("Cabin Type", selected_row["type"])
cols[3].metric("Matched Rows", int(selected_row["rows"]))
cols[4].metric("Customers", int(selected_row["customers"]))

for year in [2025, 2024]:
    if status_by_year.get(year) != "OK":
        st.warning(f"{year}: {status_by_year.get(year)}")

st.markdown("### Yearly KPI Comparison")
st.dataframe(format_yearly_kpi_table(yearly_kpi_df), use_container_width=True, hide_index=True)

st.markdown("### Monthly Loss % Comparison — 2024 / 2025 / 2026")
st.dataframe(format_loss_comparison_table(loss_compare_df), use_container_width=True, hide_index=True)

st.markdown("### Loss % Trend")
st.plotly_chart(build_multi_year_loss_chart(loss_by_year), use_container_width=True)

st.markdown("### Sale vs Total — 2026")
st.plotly_chart(build_sale_total_chart(summary_by_year[2026], 2026), use_container_width=True)

st.markdown("### Summary Tables — Same Place, No Separate Tabs")
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

# ---------------------------------------------------------
# Lightweight export
# ---------------------------------------------------------
st.subheader("Printable Export — Selected Cabin Only")
st.caption(
    "This export creates one readable report for the selected cabin, plus CSV support files. "
    "It avoids generating one PNG per ranked cabin, which is what makes hosted Streamlit apps slow or stuck."
)

report_key = f"{province}|{ranking_month}|{selected_cabin_key}|{','.join(map(str, sorted(summary_by_year.keys())))}"

if "report_cache_key" not in st.session_state:
    st.session_state["report_cache_key"] = None
if "report_png_bytes" not in st.session_state:
    st.session_state["report_png_bytes"] = None
if "report_zip_bytes" not in st.session_state:
    st.session_state["report_zip_bytes"] = None

if st.button("Build selected cabin printable report", type="primary"):
    with st.spinner("Building one clean printable report..."):
        report_png = make_printable_selected_report_png_bytes(
            province=province,
            cabin_name=resolved_name,
            cabin_type=str(selected_row["type"]),
            ranking_month=ranking_month,
            summary_by_year=summary_by_year,
            loss_by_year=loss_by_year,
            yearly_kpi_df=yearly_kpi_df,
            loss_compare_df=loss_compare_df,
        )
        report_zip = build_selected_report_zip_bytes(
            province=province,
            cabin_name=resolved_name,
            ranking_month=ranking_month,
            report_png_bytes=report_png,
            visible_ranking_df=visible_ranking_df,
            summary_by_year=summary_by_year,
            loss_compare_df=loss_compare_df,
            yearly_kpi_df=yearly_kpi_df,
        )
        st.session_state["report_cache_key"] = report_key
        st.session_state["report_png_bytes"] = report_png
        st.session_state["report_zip_bytes"] = report_zip

if st.session_state.get("report_cache_key") == report_key and st.session_state.get("report_png_bytes"):
    safe_cabin_name = safe_filename(resolved_name)
    st.success("Report ready.")
    col_png, col_zip = st.columns(2)
    with col_png:
        st.download_button(
            "Download selected cabin report PNG",
            data=st.session_state["report_png_bytes"],
            file_name=safe_filename(f"{province}_Cabin_{safe_cabin_name}_{ranking_month}_{LATEST_YEAR}_printable_report.png"),
            mime="image/png",
        )
    with col_zip:
        st.download_button(
            "Download ZIP: report PNG + CSV files",
            data=st.session_state["report_zip_bytes"],
            file_name=safe_filename(f"{province}_Cabin_{safe_cabin_name}_{ranking_month}_{LATEST_YEAR}_report_package.zip"),
            mime="application/zip",
        )

with st.expander("Why batch PNG export was removed"):
    st.write(
        "Generating one tall PNG for every ranked cabin is expensive in memory and CPU. "
        "On Streamlit Cloud, the download data is kept in memory while the user is connected, "
        "so a large ZIP can easily make the app slow or unstable. Use CSV for batch ranking and PNG for one selected cabin report."
    )
