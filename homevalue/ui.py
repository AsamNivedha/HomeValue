"""Shared Streamlit UI helpers: styling, layout components and widget wrappers."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import config as cfg
from .charts import PLOT_CONFIG
from .cleaning import QualityIssue
from .errors import HomeValueError

esc = html.escape

CSS = """
<style>
:root{
  --hv-primary:#2B59C3; --hv-primary-soft:#EAF0FC; --hv-ink:#111827; --hv-body:#374151;
  --hv-muted:#6B7280; --hv-line:#E5E7EB; --hv-soft:#F6F8FB;
  --hv-warn:#92400E; --hv-warn-bg:#FEF6E7; --hv-ok:#166534; --hv-ok-bg:#EDF8F0;
  --hv-err:#991B1B; --hv-err-bg:#FDEEEE;
}
html, body, [class*="css"], .stApp {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp { background:#FFFFFF; color:var(--hv-body); }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:4rem; }
h1,h2,h3 { color:var(--hv-ink); letter-spacing:-0.01em; }
.hv-brand { font-size:1.15rem; font-weight:700; color:var(--hv-ink); letter-spacing:-0.01em; }
.hv-brand span { color:var(--hv-primary); }
.hv-hero { padding:3.5rem 0 1.5rem 0; }
.hv-hero h1 { font-size:clamp(2.2rem,5vw,3.6rem); line-height:1.08; margin:0.3rem 0 1rem 0; font-weight:750; }
.hv-hero .hv-lede { font-size:1.12rem; line-height:1.6; max-width:46rem; color:var(--hv-body); }
.hv-eyebrow { text-transform:uppercase; font-size:.78rem; letter-spacing:.12em; color:var(--hv-primary); font-weight:700; }
.hv-steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; margin:1.5rem 0 1rem 0; }
.hv-step { border:1px solid var(--hv-line); border-radius:12px; padding:1.1rem 1.2rem; background:#fff; }
.hv-step .n { font-size:.8rem; font-weight:700; color:var(--hv-primary); }
.hv-step .t { font-size:1.1rem; font-weight:700; color:var(--hv-ink); margin:.15rem 0 .3rem 0; }
.hv-step .d { font-size:.95rem; color:var(--hv-body); line-height:1.5; }
.hv-context { display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; align-items:baseline; border:1px solid var(--hv-line);
  background:var(--hv-soft); border-radius:12px; padding:.75rem 1rem; margin:.5rem 0 1rem 0; }
.hv-context .f { font-weight:700; color:var(--hv-ink); }
.hv-context .m { color:var(--hv-body); font-size:.95rem; }
.hv-section { margin:2rem 0 .6rem 0; }
.hv-section h2 { font-size:1.35rem; margin:0; font-weight:700; }
.hv-section p { margin:.25rem 0 0 0; color:var(--hv-muted); font-size:.98rem; }
.hv-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.8rem; margin:.8rem 0; }
.hv-metric { border:1px solid var(--hv-line); border-radius:12px; padding:.9rem 1rem; background:#fff; }
.hv-metric .l { font-size:.8rem; color:var(--hv-muted); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.hv-metric .v { font-size:1.6rem; font-weight:720; color:var(--hv-ink); margin-top:.15rem; line-height:1.2; }
.hv-metric .n { font-size:.85rem; color:var(--hv-body); margin-top:.35rem; line-height:1.4; }
.hv-callout { border-radius:12px; padding:.9rem 1.1rem; margin:.7rem 0; border:1px solid var(--hv-line); line-height:1.55; }
.hv-callout .h { font-weight:700; margin-bottom:.2rem; }
.hv-callout.info { background:var(--hv-primary-soft); border-color:#CFDBF6; color:#1E3A8A; }
.hv-callout.attention { background:var(--hv-warn-bg); border-color:#F3D9A4; color:var(--hv-warn); }
.hv-callout.success { background:var(--hv-ok-bg); border-color:#BFE3C8; color:var(--hv-ok); }
.hv-callout.error { background:var(--hv-err-bg); border-color:#F2C2C2; color:var(--hv-err); }
.hv-callout ul { margin:.35rem 0 .1rem 1.1rem; padding:0; }
.hv-issue { border:1px solid var(--hv-line); border-left-width:5px; border-radius:10px; padding:.9rem 1.1rem; margin:.6rem 0; background:#fff; }
.hv-issue.attention { border-left-color:#D97706; } .hv-issue.minor { border-left-color:#9CA3AF; }
.hv-issue .tag { font-size:.75rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--hv-muted); }
.hv-issue .tag.attention { color:var(--hv-warn); }
.hv-issue .ti { font-size:1.05rem; font-weight:700; color:var(--hv-ink); margin:.1rem 0 .4rem 0; }
.hv-issue .row { font-size:.95rem; line-height:1.5; margin:.2rem 0; }
.hv-issue .row b { color:var(--hv-ink); }
.hv-flow { display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin:.6rem 0; }
.hv-flow .box { border:1px solid var(--hv-line); border-radius:10px; padding:.6rem .9rem; background:#fff; min-width:140px; flex:1 1 140px; }
.hv-flow .box.end { background:var(--hv-primary-soft); border-color:#CFDBF6; }
.hv-flow .box .c { font-size:1.2rem; font-weight:720; color:var(--hv-ink); }
.hv-flow .box .t { font-size:.82rem; color:var(--hv-muted); line-height:1.35; }
.hv-obs { margin:.2rem 0 .2rem 0; padding-left:1.2rem; } .hv-obs li { margin:.45rem 0; line-height:1.55; }
.hv-result { border:1px solid #CFDBF6; background:linear-gradient(180deg,#F5F8FE,#FFFFFF); border-radius:16px; padding:1.6rem 1.8rem; margin:1rem 0; }
.hv-result .lab { text-transform:uppercase; letter-spacing:.1em; font-size:.8rem; font-weight:700; color:var(--hv-primary); }
.hv-result .val { font-size:clamp(2.3rem,6vw,3.6rem); font-weight:760; color:var(--hv-ink); line-height:1.1; margin:.3rem 0; }
.hv-result .sub { font-size:1rem; color:var(--hv-body); font-weight:600; }
.hv-result .ctx { font-size:.95rem; color:var(--hv-muted); margin-top:.5rem; line-height:1.5; }
.hv-kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.6rem 1.2rem; margin:.4rem 0; }
.hv-kv .k { font-size:.78rem; color:var(--hv-muted); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.hv-kv .v { font-size:1.02rem; color:var(--hv-ink); font-weight:600; }
.hv-note { color:var(--hv-muted); font-size:.88rem; line-height:1.5; }
.hv-legal { color:var(--hv-muted); font-size:.82rem; border-top:1px solid var(--hv-line); margin-top:3rem; padding-top:1rem; }
div[data-testid="stForm"] { border:1px solid var(--hv-line); border-radius:14px; padding:1.2rem 1.4rem; }
.stButton>button, .stDownloadButton>button, div[data-testid="stFormSubmitButton"]>button { border-radius:10px; font-weight:600; min-height:2.6rem; }
div[data-testid="stFileUploader"] section { border-radius:12px; }
@media (max-width:640px){
  .hv-hero { padding-top:1.5rem; } .block-container { padding-left:1rem; padding-right:1rem; }
  .hv-result { padding:1.2rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def md(html_text: str) -> None:
    """Render an HTML snippet. Snippets must contain no blank lines or indentation."""
    st.markdown(html_text, unsafe_allow_html=True)


# ------------------------------------------------------------------ width-compat wrappers
def _version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (0, 0)


_NEW_WIDTH_API = _version() >= (1, 50)


def _stretch(fn, *args, **kwargs):
    if _NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except TypeError:
            pass
    return fn(*args, use_container_width=True, **kwargs)


def button(label: str, **kwargs) -> bool:
    return _stretch(st.button, label, **kwargs)


def download(label: str, data, file_name: str, mime: str, key: str, **kwargs) -> None:
    _stretch(st.download_button, label, data=data, file_name=file_name, mime=mime, key=key, **kwargs)


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type object columns to text so the table can always be rendered."""
    out = df
    for col in df.columns:
        series = df[col]
        if series.dtype == object and pd.api.types.infer_dtype(series, skipna=True) not in ("string", "empty"):
            if out is df:
                out = df.copy()
            out[col] = series.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def table(df: pd.DataFrame, **kwargs) -> None:
    _stretch(st.dataframe, _arrow_safe(df), **kwargs)


def plot(fig, key: str) -> None:
    _stretch(st.plotly_chart, fig, key=key, config=PLOT_CONFIG, theme=None)


def segmented(label: str, options: list[str], key: str) -> str:
    """A pill-style selector that remembers its choice and cannot end up empty."""
    mem = f"_last_{key}"
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = st.session_state.get(mem, options[0])
        if st.session_state[key] not in options:
            st.session_state[key] = options[0]
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control(label, options, key=key, label_visibility="collapsed")
    else:
        choice = st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed")
    if choice is None:
        choice = st.session_state.get(mem, options[0])
    st.session_state[mem] = choice
    return choice


# ------------------------------------------------------------------ components
def brand() -> None:
    md('<div class="hv-brand">Home<span>Value</span></div>')


def section(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    md(f'<div class="hv-section"><h2>{esc(title)}</h2>{sub}</div>')


def metric_grid(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="hv-metric"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
        + (f'<div class="n">{esc(note)}</div>' if note else "")
        + "</div>"
        for label, value, note in items
    )
    md(f'<div class="hv-metrics">{cells}</div>')


_CALLOUT_LABEL = {"info": "Note", "attention": "▲ Needs attention", "success": "✓ All clear", "error": "✕ Problem"}


def callout(kind: str, title: str, body: str = "") -> None:
    """body is plain text (escaped here)."""
    head = esc(title or _CALLOUT_LABEL[kind])
    text = f"<div>{esc(body)}</div>" if body else ""
    md(f'<div class="hv-callout {kind}"><div class="h">{head}</div>{text}</div>')


def error_panel(err: HomeValueError) -> None:
    expected = ""
    if err.expected:
        items = "".join(f"<li>{esc(line)}</li>" for line in err.expected)
        expected = f'<div class="h" style="margin-top:.7rem">What HomeValue expected</div><ul>{items}</ul>'
    action = (
        f'<div class="h" style="margin-top:.7rem">What you can do</div><div>{esc(err.action)}</div>' if err.action else ""
    )
    md(
        f'<div class="hv-callout error"><div class="h">✕ {esc(err.title)}</div>'
        f'<div class="h" style="margin-top:.5rem">What went wrong</div><div>{esc(err.what)}</div>{expected}{action}</div>'
    )


def issue_card(issue: QualityIssue) -> None:
    label = "▲ Needs attention" if issue.severity == "attention" else "● Minor note"
    md(
        f'<div class="hv-issue {issue.severity}"><div class="tag {issue.severity}">{label}</div>'
        f'<div class="ti">{esc(issue.title)}</div>'
        f'<div class="row"><b>Impact.</b> {esc(issue.impact)}</div>'
        f'<div class="row"><b>HomeValue action.</b> {esc(issue.action)}</div></div>'
    )


def flow(funnel: list[tuple[str, int]]) -> None:
    boxes = []
    for i, (label, count) in enumerate(funnel):
        if 0 < i < len(funnel) - 1 and count == 0:
            continue
        sign = "−" if 0 < i < len(funnel) - 1 else ""
        cls = "box end" if i == len(funnel) - 1 else "box"
        boxes.append(f'<div class="{cls}"><div class="c">{sign}{count:,}</div><div class="t">{esc(label)}</div></div>')
    md('<div class="hv-flow">' + "".join(boxes) + "</div>")


def key_values(pairs: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in pairs)
    md(f'<div class="hv-kv">{cells}</div>')


def bullets(items: list[str]) -> None:
    md('<ul class="hv-obs">' + "".join(f"<li>{esc(t)}</li>" for t in items) + "</ul>")


def note(text: str) -> None:
    md(f'<div class="hv-note">{esc(text)}</div>')


def legal_footer() -> None:
    md(f'<div class="hv-legal">{esc(cfg.DISCLAIMER)}</div>')


def goto(page: str, **extra) -> None:
    """Callback helper: switch page (and optionally sub-view keys)."""
    st.session_state["nav_page"] = page
    for key, value in extra.items():
        st.session_state[key] = value


def callout_list(kind: str, title: str, items: list[str]) -> None:
    lis = "".join(f"<li>{esc(t)}</li>" for t in items)
    md(f'<div class="hv-callout {kind}"><div class="h">{esc(title)}</div><ul>{lis}</ul></div>')


def submit(label: str, **kwargs) -> bool:
    return _stretch(st.form_submit_button, label, **kwargs)
