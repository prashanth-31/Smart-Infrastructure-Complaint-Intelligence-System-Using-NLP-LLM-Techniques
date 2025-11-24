from __future__ import annotations

import html
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config import DEFAULT_DATASET
from models.pipeline_loader import StubModelWarning, load_model_bundle
from utils.analysis_pipeline import AnalysisResult, analyze_complaint
from utils.data_store import append_analysis, load_dataset
from utils.preprocessing import build_feature_row
from utils.visualization import (
    build_wordcloud_image,
    complaints_over_time,
    issue_type_bar,
    latest_complaints_table,
    severity_pie,
    severity_urgency_matrix,
    urgency_timeline_area,
)

st.set_page_config(
    page_title="Smart Infrastructure Complaint Intelligence",
    page_icon="📊",
    layout="wide",
)

# Capture stub warnings during model loading
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", StubModelWarning)
    bundle = load_model_bundle(
        enable_stubs=True,  # Set to True to use stub models when real models fail to load
    )
    # Store warnings for display in UI
    stub_warnings = [warning for warning in w if issubclass(warning.category, StubModelWarning)]

APP_ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def _load_dataset_cached(dataset_path: str) -> pd.DataFrame:
    return load_dataset(Path(dataset_path))


def _render_global_styles(theme: str) -> None:
    css_path = APP_ROOT / "assets" / "css" / "app.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

    shared_transitions = """
.stApp, html, body, .insight-card, .history-card, .hero-card, .metric-highlight, .summary-pill, .entity-chip {
    transition: background-color 0.6s ease, color 0.6s ease, border-color 0.6s ease, box-shadow 0.6s ease;
}
.insight-card, .history-card {
    backdrop-filter: blur(14px);
    border-radius: 14px;
}
.analysis-inline-ner {
    padding: 12px 16px;
    border-radius: 12px;
    font-family: "Space Mono", "Fira Code", monospace;
    font-size: 0.92rem;
    line-height: 1.7;
    transition: background-color 0.6s ease, color 0.6s ease, border-color 0.6s ease;
}
"""

    if theme == "dark":
        st.markdown(
            f"""
<style>
html, body, .stApp {{
    background: radial-gradient(circle at top, #1e293b 0%, #0f172a 50%, #020617 100%) !important;
    color: #f1f5f9 !important;
}}

:root {{
    --hero-gradient: linear-gradient(135deg, #1e3a8a, #2563eb, #3b82f6);
    --card-bg: rgba(30, 41, 59, 0.95);
    --card-border: rgba(148, 163, 184, 0.25);
    --shadow-soft: 0 24px 65px rgba(0, 0, 0, 0.7);
    --shadow-card: 0 20px 50px rgba(0, 0, 0, 0.6);
    --text-muted: #94a3b8;
    --text-strong: #f1f5f9;
    --text-primary: #e2e8f0;
    --bg-entity: rgba(79, 70, 229, 0.25);
    --bg-highlight: rgba(30, 41, 59, 0.95);
    --border-entity: rgba(129, 140, 248, 0.4);
}}

.hero-card {{
    color: #ffffff;
}}

.hero-card__eyebrow,
.hero-card__title,
.hero-card__subtitle {{
    color: #ffffff !important;
}}

.metric-highlight {{
    background: rgba(30, 41, 59, 0.95);
    border-color: rgba(148, 163, 184, 0.3);
}}

.metric-highlight h3 {{
    color: #f1f5f9 !important;
}}

.metric-highlight span {{
    color: #94a3b8 !important;
}}

.info-banner {{
    background: rgba(30, 58, 138, 0.35);
    border-color: rgba(96, 165, 250, 0.4);
    color: #bfdbfe !important;
}}

.insight-card {{
    background: rgba(30, 41, 59, 0.95);
    border-color: rgba(148, 163, 184, 0.25);
}}

.insight-card h4 {{
    color: #f1f5f9 !important;
}}

.insight-card ul li,
.insight-card p {{
    color: #cbd5e1 !important;
}}

.summary-pill {{
    background: rgba(59, 130, 246, 0.25);
    border-color: rgba(96, 165, 250, 0.45);
    color: #bfdbfe !important;
}}

.summary-pill--severity-high {{
    background: rgba(239, 68, 68, 0.25);
    border-color: rgba(248, 113, 113, 0.45);
    color: #fecaca !important;
}}

.summary-pill--severity-medium {{
    background: rgba(234, 179, 8, 0.25);
    border-color: rgba(250, 204, 21, 0.45);
    color: #fef08a !important;
}}

.summary-pill--severity-low {{
    background: rgba(34, 197, 94, 0.25);
    border-color: rgba(74, 222, 128, 0.45);
    color: #bbf7d0 !important;
}}

.summary-pill--urgency-urgent {{
    background: rgba(220, 38, 38, 0.25);
    border-color: rgba(248, 113, 113, 0.45);
    color: #fecaca !important;
}}

.summary-pill--urgency-concerned {{
    background: rgba(59, 130, 246, 0.25);
    border-color: rgba(96, 165, 250, 0.45);
    color: #bfdbfe !important;
}}

.summary-pill--urgency-neutral {{
    background: rgba(124, 58, 237, 0.25);
    border-color: rgba(167, 139, 250, 0.45);
    color: #ddd6fe !important;
}}

.badge--severity-high {{
    background: rgba(239, 68, 68, 0.25);
    border-color: rgba(248, 113, 113, 0.4);
    color: #fca5a5 !important;
}}

.badge--severity-medium {{
    background: rgba(234, 179, 8, 0.25);
    border-color: rgba(250, 204, 21, 0.4);
    color: #fde047 !important;
}}

.badge--severity-low {{
    background: rgba(34, 197, 94, 0.25);
    border-color: rgba(74, 222, 128, 0.4);
    color: #86efac !important;
}}

.badge--urgency-urgent {{
    background: rgba(220, 38, 38, 0.25);
    border-color: rgba(248, 113, 113, 0.4);
    color: #fca5a5 !important;
}}

.badge--urgency-concerned {{
    background: rgba(59, 130, 246, 0.25);
    border-color: rgba(96, 165, 250, 0.4);
    color: #93c5fd !important;
}}

.badge--urgency-neutral {{
    background: rgba(124, 58, 237, 0.25);
    border-color: rgba(167, 139, 250, 0.4);
    color: #c4b5fd !important;
}}

.entity-chip {{
    background: rgba(79, 70, 229, 0.25);
    border-color: rgba(129, 140, 248, 0.4);
    color: #c7d2fe !important;
}}

.entity-chip small {{
    color: #a5b4fc !important;
}}

.history-card {{
    background: rgba(30, 41, 59, 0.95);
    border-color: rgba(148, 163, 184, 0.25);
}}

.history-card__headline {{
    color: #f1f5f9 !important;
}}

.muted-text {{
    color: #94a3b8 !important;
}}

.analysis-inline-ner {{
    background: rgba(30, 58, 138, 0.35);
    border: 1px dashed rgba(96, 165, 250, 0.45);
    color: #bfdbfe !important;
}}

.download-pill {{
    border-color: rgba(96, 165, 250, 0.5);
    background: rgba(59, 130, 246, 0.25);
    color: #93c5fd !important;
}}

{shared_transitions}
</style>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<style>
html, body, .stApp {{
    background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 40%, #e0e7ff 100%) !important;
    color: #0f172a !important;
}}

:root {{
    --hero-gradient: linear-gradient(135deg, #1e40af, #2563eb, #3b82f6);
    --card-bg: rgba(255, 255, 255, 0.96);
    --card-border: rgba(226, 232, 240, 0.9);
    --shadow-soft: 0 24px 55px rgba(15, 23, 42, 0.12);
    --shadow-card: 0 20px 48px rgba(15, 23, 42, 0.09);
    --text-muted: #475569;
    --text-strong: #0f172a;
    --text-primary: #1e293b;
    --bg-entity: #eef2ff;
    --bg-highlight: #ffffff;
    --border-entity: rgba(79, 70, 229, 0.35);
}}

.hero-card {{
    color: #ffffff;
}}

.hero-card__eyebrow,
.hero-card__title,
.hero-card__subtitle {{
    color: #ffffff !important;
}}

.metric-highlight {{
    background: rgba(255, 255, 255, 0.96);
    border-color: rgba(226, 232, 240, 0.9);
}}

.metric-highlight h3 {{
    color: #0f172a !important;
}}

.metric-highlight span {{
    color: #475569 !important;
}}

.info-banner {{
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af !important;
}}

.insight-card {{
    background: rgba(255, 255, 255, 0.96);
    border-color: rgba(226, 232, 240, 0.9);
}}

.insight-card h4 {{
    color: #0f172a !important;
}}

.insight-card ul li,
.insight-card p {{
    color: #334155 !important;
}}

.summary-pill {{
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.35);
    color: #1e40af !important;
}}

.summary-pill--severity-high {{
    background: rgba(239, 68, 68, 0.18);
    border-color: rgba(239, 68, 68, 0.4);
    color: #b91c1c !important;
}}

.summary-pill--severity-medium {{
    background: rgba(234, 179, 8, 0.25);
    border-color: rgba(234, 179, 8, 0.45);
    color: #92400e !important;
}}

.summary-pill--severity-low {{
    background: rgba(34, 197, 94, 0.20);
    border-color: rgba(34, 197, 94, 0.45);
    color: #15803d !important;
}}

.summary-pill--urgency-urgent {{
    background: rgba(220, 38, 38, 0.18);
    border-color: rgba(220, 38, 38, 0.4);
    color: #991b1b !important;
}}

.summary-pill--urgency-concerned {{
    background: rgba(59, 130, 246, 0.18);
    border-color: rgba(59, 130, 246, 0.38);
    color: #1e40af !important;
}}

.summary-pill--urgency-neutral {{
    background: rgba(124, 58, 237, 0.20);
    border-color: rgba(124, 58, 237, 0.4);
    color: #6b21a8 !important;
}}

.badge--severity-high {{
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.3);
    color: #b91c1c !important;
}}

.badge--severity-medium {{
    background: rgba(234, 179, 8, 0.20);
    border-color: rgba(234, 179, 8, 0.35);
    color: #92400e !important;
}}

.badge--severity-low {{
    background: rgba(34, 197, 94, 0.18);
    border-color: rgba(34, 197, 94, 0.35);
    color: #15803d !important;
}}

.badge--urgency-urgent {{
    background: rgba(220, 38, 38, 0.15);
    border-color: rgba(220, 38, 38, 0.3);
    color: #991b1b !important;
}}

.badge--urgency-concerned {{
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.3);
    color: #1e40af !important;
}}

.badge--urgency-neutral {{
    background: rgba(124, 58, 237, 0.18);
    border-color: rgba(124, 58, 237, 0.35);
    color: #6b21a8 !important;
}}

.entity-chip {{
    background: #eef2ff;
    border-color: rgba(79, 70, 229, 0.35);
    color: #3730a3 !important;
}}

.entity-chip small {{
    color: #6366f1 !important;
}}

.history-card {{
    background: rgba(255, 255, 255, 0.96);
    border-color: rgba(226, 232, 240, 0.9);
}}

.history-card__headline {{
    color: #0f172a !important;
}}

.muted-text {{
    color: #475569 !important;
}}

.analysis-inline-ner {{
    background: rgba(191, 219, 254, 0.35);
    border: 1px dashed rgba(96, 165, 250, 0.5);
    color: #1e40af !important;
}}

.download-pill {{
    border-color: #2563eb;
    background: rgba(37, 99, 235, 0.08);
    color: #1e40af !important;
}}

{shared_transitions}
</style>
""",
            unsafe_allow_html=True,
        )


def _calculate_summary(df: pd.DataFrame, components: Dict[str, str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": int(df.shape[0]),
        "issue_types": 0,
        "today_count": 0,
        "urgent_pct": 0.0,
        "top_issue": "",
        "top_location": "",
        "last_update": None,
        "live_components": sum(1 for status in components.values() if status == "live"),
        "component_total": len(components),
    }
    if df.empty:
        return summary

    working = df.copy()
    if "created_at" in working.columns:
        working["_created_at_dt"] = pd.to_datetime(working["created_at"], errors="coerce")
        latest = working["_created_at_dt"].max()
        summary["last_update"] = latest
        today_start = pd.Timestamp.utcnow().normalize()
        summary["today_count"] = int((working["_created_at_dt"] >= today_start).sum())

    if "issue_type" in working.columns and not working["issue_type"].dropna().empty:
        summary["issue_types"] = int(working["issue_type"].nunique())
        summary["top_issue"] = str(working["issue_type"].mode().iloc[0])

    if "location" in working.columns and not working["location"].dropna().empty:
        summary["top_location"] = str(working["location"].mode().iloc[0])

    if "urgency" in working.columns and summary["total"]:
        urgent_total = (working["urgency"].astype(str).str.lower() == "urgent").sum()
        summary["urgent_pct"] = round((urgent_total / summary["total"]) * 100, 1)

    return summary


def _format_timestamp(ts: Any) -> str:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        return "—"
    return ts.strftime("%d %b %Y • %H:%M")


def _component_badges(components: Dict[str, str]) -> str:
    badges = []
    for key, status in components.items():
        label = key.replace("_", " ").title()
        cls = "status-tag--live" if status == "live" else "status-tag--stub"
        badges.append(f"<span class='status-tag {cls}'>{label}</span>")
    return "".join(badges)


def _generate_recommendations(analysis: AnalysisResult) -> List[str]:
    severity = analysis.severity.lower()
    urgency = analysis.urgency.lower()
    recs: List[str] = []
    if severity == "high":
        recs.append("Escalate to the rapid response team and notify field operations immediately.")
    if urgency in {"angry/urgent", "urgent"}:
        recs.append("Prioritise dispatch within 2 hours and issue a citizen acknowledgment SMS.")
    if "road" in analysis.issue_type.lower():
        recs.append("Coordinate with traffic management for temporary safety barricades.")
    if not recs:
        recs.append("Queue for scheduled maintenance after verifying resource availability.")
    return recs


def _entity_badges(entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return ""
    parts = []
    for ent in entities:
        text = html.escape(str(ent.get("text", ""))).strip()
        label = html.escape(str(ent.get("label", ""))).strip()
        if not text:
            continue
        formatted = f"{text}({label})" if label else text
        parts.append(formatted)
    return ", ".join(parts)


def _token_annotations(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not text:
        return []
    spans = [
        (int(ent.get("start", 0) or 0), int(ent.get("end", ent.get("start", 0)) or 0), str(ent.get("label", "")))
        for ent in entities
    ]
    tokens: List[Dict[str, str]] = []
    for match in re.finditer(r"\S+", text):
        start, end = match.start(), match.end()
        label = "O"
        for span_start, span_end, span_label in spans:
            if start >= span_start and end <= span_end:
                label = span_label or "O"
                break
        tokens.append({"token": match.group(), "label": label})
    return tokens


def _init_session_state() -> None:
    if "dataset" not in st.session_state:
        st.session_state.dataset = _load_dataset_cached(str(DEFAULT_DATASET))
    if "history" not in st.session_state:
        st.session_state.history = []
    if "complaint_input" not in st.session_state:
        st.session_state.complaint_input = ""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def _highlight_entities(text: str, entities: List[Dict[str, str]]) -> str:
    if not entities:
        return html.escape(text)
    
    # Use theme-aware colors with good contrast
    color_map = {
        "LOCATION": "rgba(253, 224, 71, 0.4)",  # Yellow with transparency
        "LOC": "rgba(253, 224, 71, 0.4)",
        "GPE": "rgba(253, 224, 71, 0.4)",
        "ORG": "rgba(96, 165, 250, 0.35)",  # Blue with transparency
        "PROBLEM": "rgba(248, 113, 113, 0.35)",  # Red with transparency
    }
    
    result: List[str] = []
    cursor = 0
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
    for ent in sorted_entities:
        start = int(ent.get("start", 0) or 0)
        end = int(ent.get("end", start) or start)
        if end < start:
            end = start
        label = ent.get("label", "")
        if start > cursor:
            result.append(html.escape(text[cursor:start]))
        color = color_map.get(label.upper(), "rgba(167, 139, 250, 0.35)")
        ent_text = html.escape(text[start:end])
        span = (
            f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px; "
            f"font-weight:600; border:1px solid rgba(0,0,0,0.1);'>"
            f"{ent_text} <small style='opacity:0.8;'>({html.escape(label)})</small></span>"
        )
        result.append(span)
        cursor = end
    result.append(html.escape(text[cursor:]))
    return "".join(result)


def _render_sidebar(summary: Dict[str, Any], components: Dict[str, str], mode: str) -> tuple[str, str]:
    st.sidebar.title("Smart Infrastructure AI")
    st.sidebar.markdown("Real-time NLP triage for civic infrastructure resilience.")

    current_theme = st.session_state.get("theme", "light")
    dark_enabled = st.sidebar.toggle(
        "Dark mode",
        value=current_theme == "dark",
        key="dark_mode_toggle",
    )
    theme = "dark" if dark_enabled else "light"
    st.session_state.theme = theme

    st.sidebar.markdown("### Snapshot")
    st.sidebar.metric("Complaints", f"{summary['total']:,}", delta=f"+{summary['today_count']} today")
    st.sidebar.metric("Urgent Share", f"{summary['urgent_pct']}%")
    st.sidebar.markdown(f"Last update: **{_format_timestamp(summary['last_update'])}**")

    component_badges = _component_badges(components)
    if component_badges:
        st.sidebar.markdown("### Pipeline Components")
        st.sidebar.markdown(component_badges, unsafe_allow_html=True)

    st.sidebar.markdown("### Resources")
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/3/3f/SDG_logo.png",
        caption="SDG 9: Industry, Innovation and Infrastructure",
        use_container_width=True,
    )
    st.sidebar.markdown(
        """
- Multi-Task BERT (31 categories, severity, urgency)
- spaCy NER for civic entities
- Real-time complaint classification
"""
    )

    csv_data = st.session_state.dataset.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download analytics dataset",
        data=csv_data,
        file_name="complaints_dataset.csv",
        mime="text/csv",
    )

    st.sidebar.markdown("---")
    selected = st.sidebar.radio(
        "Navigate",
        ["Complaint Analyzer", "Dashboard & Analytics", "About"],
        key="sidebar_nav",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span class='muted-text'>Pipeline mode:</span> "
        f"<strong>{mode.upper()}</strong>",
        unsafe_allow_html=True,
    )
    return selected, theme


def _render_header(metadata: Dict[str, Any], summary: Dict[str, Any]) -> None:
    mode = metadata.get("mode", "production")
    badge = "LIVE" if mode == "production" else "STUB"
    badge_class = "status-tag--live" if badge == "LIVE" else "status-tag--stub"
    components = metadata.get("components", {})
    component_badges = _component_badges(components)
    last_update = _format_timestamp(summary.get("last_update"))

    st.markdown(
        """
<div class="hero-card">
  <div class="hero-card__eyebrow">Pipeline status</div>
  <div class="hero-card__title">Smart Infrastructure Complaint Intelligence</div>
  <p class="hero-card__subtitle">City-scale complaint intelligence powered by Multi-Task BERT for category, severity, and urgency prediction with spaCy NER in a unified Streamlit dashboard.</p>
  <div class="hero-card__status">
    <span class="status-tag {badge_class}">Mode: {badge}</span>{component_badges}
  </div>
  <p class="hero-card__subtitle" style="margin-top:18px;">Last dataset update • {last_update}</p>
</div>
""".format(badge_class=badge_class, badge=badge, component_badges=component_badges, last_update=last_update),
        unsafe_allow_html=True,
    )

    metrics = [
        ("Total complaints", f"{summary['total']:,}", f"+{summary['today_count']} today"),
        ("Issue categories", summary.get("issue_types", 0), summary.get("top_issue", "") or "—"),
        ("Urgent share", f"{summary.get('urgent_pct', 0.0)}%", ""),
        (
            "Live components",
            f"{summary.get('live_components', 0)}/{summary.get('component_total', 0)}",
            summary.get("top_location", "") or "—",
        ),
    ]
    cols = st.columns(len(metrics))
    for col, (title, value, subtitle) in zip(cols, metrics):
        col.markdown(
            """
<div class="metric-highlight">
  <span>{title}</span>
  <h3>{value}</h3>
  <div class="muted-text">{subtitle}</div>
</div>
""".format(title=title, value=value, subtitle=subtitle or ""),
            unsafe_allow_html=True,
        )

    if mode != "production":
        # Show detailed stub warning if components are using stubs
        stub_components = bundle.get_stub_components()
        if stub_components:
            st.error(
                f"⚠️ **WARNING: STUB MODELS IN USE**\\n\\n"
                f"The following components are NOT providing ML-based predictions:\\n"
                f"- {', '.join(stub_components)}\\n\\n"
                f"**Impact:** Predictions will be random or heuristic-based.\\n"
                f"**Action:** Check model files and loading errors in logs.\\n\\n"
                f"**DO NOT use in production without fixing model loading issues.**"
            )
        else:
            st.warning("Pipeline mode indicates non-production configuration.")


def _handle_analysis_page() -> None:
    st.subheader("Complaint Analyzer")
    st.markdown(
        "<div class='info-banner'>💡 Tip: Mention landmarks, ward names, and observed hazards for richer entity extraction.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="insight-card">
    <h4>What this AI does best</h4>
    <ul>
        <li>Classifies complaints into 31 infrastructure categories using Multi-Task BERT with civic keyword heuristics.</li>
        <li>Predicts severity levels (LOW, MEDIUM, HIGH) with context-aware deep learning.</li>
        <li>Evaluates urgency (NEUTRAL, CONCERNED, URGENT) through multi-head attention mechanisms.</li>
        <li>Extracts locations, assets, and problem terms with spaCy NER for precise GIS routing.</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    samples = {
        "Street Light Outage": "The street light near the 5th cross in Jayanagar has not been working for three days, leaving the stretch dark and prone to accidents.",
        "Water Supply Disruption": "Residents of Indiranagar 9th Main have had no water supply for 24 hours. Overhead tanks are empty and senior citizens are impacted.",
        "Garbage Overflow": "Garbage heap near Rajajinagar 4th block park has not been cleared for a week. Stray dogs and foul smell are troubling residents.",
    }

    sample_choice = st.selectbox(
        "Need a sample complaint?",
        ["Custom input"] + list(samples.keys()),
        key="sample_selector",
        help="Load a prototype complaint to explore the pipeline.",
    )
    if sample_choice != "Custom input":
        st.session_state.complaint_input = samples[sample_choice]

    complaint_text = st.text_area(
        "Describe the complaint",
        key="complaint_input",
        height=220,
        placeholder="Example: There is a gaping pothole near ...",
        help="Include location cues, issue details, severity hints, and urgency signals.",
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        persist = st.checkbox("Add result to analytics dataset", value=True)
    with col_right:
        run_button = st.button("Analyze Complaint", type="primary", use_container_width=True)
        clear_button = st.button("Clear", use_container_width=True)

    if clear_button:
        st.session_state.complaint_input = ""
        st.rerun()

    if run_button and complaint_text.strip():
        with st.spinner("Running NLP pipeline..."):
            analysis = analyze_complaint(complaint_text.strip(), bundle)
        st.session_state.history.append(analysis)
        if persist:
            row = build_feature_row(
                {
                    "text": analysis.raw_text,
                    "issue_type": analysis.issue_type,
                    "severity": analysis.severity,
                    "urgency": analysis.urgency,
                    "location": analysis.location,
                }
            )
            row["created_at"] = pd.Timestamp.utcnow().isoformat()
            append_analysis(row, DEFAULT_DATASET)
            _load_dataset_cached.clear()
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, pd.DataFrame([row])],
                ignore_index=True,
            )

        _render_analysis_output(analysis)

    if st.session_state.history:
        st.markdown("### Recent Analyses")
        for item in st.session_state.history[::-1][:5]:
            _render_analysis_output(item, compact=True)


def _render_analysis_output(analysis: AnalysisResult, compact: bool = False) -> None:
    severity_class = {
        "high": "summary-pill--severity-high",
        "medium": "summary-pill--severity-medium",
        "low": "summary-pill--severity-low",
    }.get(analysis.severity.lower(), "summary-pill--severity-medium")
    urgency_class = {
        "angry/urgent": "summary-pill--urgency-urgent",
        "urgent": "summary-pill--urgency-urgent",
        "concerned": "summary-pill--urgency-concerned",
        "neutral": "summary-pill--urgency-neutral",
    }.get(analysis.urgency.lower(), "summary-pill--urgency-neutral")

    if compact:
        preview = analysis.raw_text.strip()
        if len(preview) > 160:
            preview = preview[:157] + "..."
        st.markdown(
            """
<div class="history-card">
  <div class="history-card__headline">
    <span>{issue}</span>
    <div>
      <span class="badge {severity_cls}">Severity: {severity}</span>
      <span class="badge {urgency_cls}">Urgency: {urgency}</span>
    </div>
  </div>
  <p class="muted-text">{preview}</p>
</div>
""".format(
                issue=html.escape(analysis.issue_type),
                severity_cls=severity_class.replace("summary-pill", "badge"),
                severity=html.escape(analysis.severity.title()),
                urgency_cls=urgency_class.replace("summary-pill", "badge"),
                urgency=html.escape(analysis.urgency.title()),
                preview=html.escape(preview),
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
<div class="insight-card">
  <h4>Classification summary</h4>
  <div class="summary-pill-row">
        <span class="summary-pill">Category: {issue}</span>
    <span class="summary-pill {severity_cls}">Severity: {severity}</span>
    <span class="summary-pill {urgency_cls}">Urgency: {urgency}</span>
  </div>
</div>
""".format(
            issue=html.escape(analysis.issue_type),
            severity_cls=severity_class,
            severity=html.escape(analysis.severity.title()),
            urgency_cls=urgency_class,
            urgency=html.escape(analysis.urgency.title()),
        ),
        unsafe_allow_html=True,
    )

    entity_html = _entity_badges(analysis.entities)
    if entity_html:
        st.markdown(
            """
<div class="insight-card">
  <h4>Detected entities</h4>
  <div class="entity-chip-row">{entities}</div>
</div>
""".format(entities=entity_html),
            unsafe_allow_html=True,
        )

    token_rows = _token_annotations(analysis.raw_text, analysis.entities)
    if token_rows:
        inline_tokens = " ".join(
            f"{html.escape(item['token'])}({html.escape(item['label'])})" for item in token_rows
        )
        st.markdown(
            """
<div class="insight-card">
  <h4>Token-Level Entities</h4>
  <div class="analysis-inline-ner">{tokens}</div>
</div>
""".format(tokens=inline_tokens),
            unsafe_allow_html=True,
        )

    highlighted = _highlight_entities(analysis.raw_text, analysis.entities)
    st.markdown(
        """
<div class="insight-card">
  <h4>Complaint context</h4>
  <div style="background:var(--bg-highlight); border-radius:10px; padding:16px; border:1px solid var(--card-border); color:var(--text-primary);">{text}</div>
</div>
""".format(text=highlighted),
        unsafe_allow_html=True,
    )

    recommendations = _generate_recommendations(analysis)
    if recommendations:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in recommendations)
        st.markdown(
            """
<div class="insight-card">
  <h4>Operational recommendations</h4>
  <ul>{items}</ul>
</div>
""".format(items=items),
            unsafe_allow_html=True,
        )


def _render_dashboard_page() -> None:
    st.subheader("Dashboard & Analytics")
    dataset = st.session_state.dataset
    if dataset.empty:
        st.warning("No complaints recorded yet. Analyze a complaint and opt to add it to the dataset.")
        return

    st.markdown("### Distribution Snapshot")
    col1, col2 = st.columns(2)
    col1.plotly_chart(issue_type_bar(dataset), use_container_width=True)
    col2.plotly_chart(severity_pie(dataset), use_container_width=True)

    heatmap = severity_urgency_matrix(dataset)
    if getattr(heatmap, "data", None):
        st.plotly_chart(heatmap, use_container_width=True)

    line_chart = complaints_over_time(dataset)
    if line_chart is not None:
        st.altair_chart(line_chart, use_container_width=True)

    urgency_area = urgency_timeline_area(dataset)
    if getattr(urgency_area, "data", None):
        st.plotly_chart(urgency_area, use_container_width=True)

    image_bytes, _ = build_wordcloud_image(dataset)
    if image_bytes:
        st.markdown("### Keyword Cloud")
        st.image(image_bytes, use_container_width=True)

    st.markdown("### Recent Complaints")
    st.dataframe(latest_complaints_table(dataset), use_container_width=True)


def _render_about_page() -> None:
    st.subheader("About the System")
    st.markdown(
        """
**Objective**
Deliver actionable intelligence on civic infrastructure complaints in real time.

**Methodology**
1. Preprocess incoming complaint text with normalization and cleaning.
2. Classify issue category, severity, and urgency simultaneously using Multi-Task BERT.
3. Extract entities using spaCy NER (locations, problem terms, organizations).
4. Apply rule-based enhancements for edge cases and keyword matching.
5. Persist results to CSV for analytics and visualization.
6. Generate real-time insights and visualizations.

**Technologies**
- Streamlit frontend
- Multi-Task BERT (PyTorch + Transformers)
- spaCy NER (en_core_web_sm)
- Pandas, NumPy
- Plotly, Altair, WordCloud

**Model Training**
Use scripts in `training/` to fine-tune and export models. Place artifacts inside `models/` before deployment.
"""
    )


def main() -> None:
    _init_session_state()
    dataset = st.session_state.dataset
    metadata = bundle.metadata or {}
    components = metadata.get("components", {})
    mode = metadata.get("mode", "production")
    summary = _calculate_summary(dataset, components)

    selected, theme = _render_sidebar(summary, components, mode)
    _render_global_styles(theme)
    _render_header(metadata, summary)

    pages = {
        "Complaint Analyzer": _handle_analysis_page,
        "Dashboard & Analytics": _render_dashboard_page,
        "About": _render_about_page,
    }
    pages[selected]()


if __name__ == "__main__":
    main()
