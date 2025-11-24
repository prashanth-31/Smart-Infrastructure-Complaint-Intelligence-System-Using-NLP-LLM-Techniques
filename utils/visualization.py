from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from utils.analysis_pipeline import map_model_label_to_category


def _normalise_issue_categories(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=str)
    normalized = series.fillna("").astype(str)
    return normalized.map(map_model_label_to_category)


def issue_type_bar(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if "issue_type" not in df.columns or df["issue_type"].dropna().empty:
        return figure
    categories = _normalise_issue_categories(df["issue_type"])
    if categories.empty:
        return figure
    counts = categories.value_counts().reset_index()
    counts.columns = ["Issue Type", "Count"]
    figure = px.bar(
        counts,
        x="Issue Type",
        y="Count",
        color="Issue Type",
        title="Complaint Distribution by Issue Type",
    )
    return figure


def severity_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    figure = px.pie(counts, names="Severity", values="Count", title="Severity Levels")
    return figure


def complaints_over_time(df: pd.DataFrame) -> Optional[alt.Chart]:
    if "created_at" not in df.columns:
        return None
    ts = (
        df.assign(created_at=pd.to_datetime(df["created_at"]).dt.date)
        .groupby("created_at")
        .size()
        .reset_index(name="count")
    )
    if ts.empty:
        return None
    return (
        alt.Chart(ts)
        .mark_line(point=True)
        .encode(x="created_at:T", y="count:Q")
        .properties(title="Complaints Over Time")
    )


def build_wordcloud_image(df: pd.DataFrame, text_column: str = "complaint_text") -> Tuple[bytes, str]:
    text = " ".join(df[text_column].dropna().astype(str))
    if not text.strip():
        return b"", ""
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def latest_complaints_table(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    cols = [c for c in ["created_at", "issue_type", "severity", "urgency", "location", "complaint_text"] if c in df.columns]
    return df[cols].tail(limit).iloc[::-1]


def severity_urgency_matrix(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if not {"severity", "urgency"}.issubset(df.columns):
        return figure
    counts = (
        df.groupby(["severity", "urgency"])
        .size()
        .reset_index(name="count")
    )
    if counts.empty:
        return figure
    severity_order = [value for value in ["High", "Medium", "Low"] if value in counts["severity"].unique()]
    urgency_order = [value for value in ["Urgent", "Concerned", "Neutral"] if value in counts["urgency"].unique()]
    figure = px.density_heatmap(
        counts,
        x="urgency",
        y="severity",
        z="count",
        color_continuous_scale="Blues",
        title="Severity vs Urgency Hotspots",
        category_orders={"severity": severity_order, "urgency": urgency_order},
    )
    figure.update_layout(margin=dict(l=40, r=30, t=60, b=40))
    return figure


def urgency_timeline_area(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if "created_at" not in df.columns or "urgency" not in df.columns:
        return figure
    timeline = (
        df.assign(created_at=pd.to_datetime(df["created_at"], errors="coerce").dt.date)
        .dropna(subset=["created_at", "urgency"])
        .groupby(["created_at", "urgency"])
        .size()
        .reset_index(name="count")
        .sort_values("created_at")
    )
    if timeline.empty:
        return figure
    figure = px.area(
        timeline,
        x="created_at",
        y="count",
        color="urgency",
        title="Urgency Trend",
    )
    figure.update_layout(margin=dict(l=40, r=30, t=60, b=50))
    return figure
