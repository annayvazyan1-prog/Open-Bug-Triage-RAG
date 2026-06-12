# -*- coding: utf-8 -*-
"""
Open-Bug Triage RAG — Streamlit app
===================================

Run with uv:
    uv run streamlit run src/app.py
"""

import os
import streamlit as st

from triage_core import (
    OpenTicketTriageTools,
    build_llm_with_tools,
    run_agent,
    DUPLICATE_THRESHOLD,
)

st.set_page_config(page_title="Open-Bug Triage", page_icon="🐛", layout="wide")
st.title("🐛 Open-Bug Triage")
st.caption("Search the open ticket backlog: find similar tickets, spot duplicates, sweep by priority.")

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("api_key")
if not api_key:
    st.info("Set OPENAI_API_KEY in a .env file, or paste it below for this session.")
    entered = st.text_input("OpenAI API key", type="password")
    if entered:
        st.session_state["api_key"] = entered
        os.environ["OPENAI_API_KEY"] = entered
        st.rerun()
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key


# ---------------------------------------------------------------------------
# Load tools + agent once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing the ticket backlog…")
def load_engine():
    tm = OpenTicketTriageTools()
    tools = tm.get_tools()
    llm_with_tools = build_llm_with_tools(tools)
    return tm, tools, llm_with_tools


try:
    tm, tools, llm_with_tools = load_engine()
except Exception as exc:  # surface init errors (bad key, missing data, etc.)
    st.error(f"Could not start the engine: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: backlog overview + duplicate threshold
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Backlog")
    s = tm.stats()
    st.metric("Open tickets", s["total"])

    st.subheader("By priority")
    st.dataframe(
        [{"Priority": k, "Count": v} for k, v in
         sorted(s["by_priority"].items(), key=lambda x: x[1], reverse=True)],
        hide_index=True, use_container_width=True,
    )
    st.subheader("By category")
    st.dataframe(
        [{"Category": k, "Count": v} for k, v in
         sorted(s["by_category"].items(), key=lambda x: x[1], reverse=True)],
        hide_index=True, use_container_width=True,
    )

    st.divider()
    threshold = st.slider(
        "Duplicate similarity cutoff", 0.50, 0.95, float(DUPLICATE_THRESHOLD), 0.01,
        help="Higher = stricter. Only matches at or above this score count as duplicates.",
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
ask_tab, similar_tab, dup_tab, browse_tab = st.tabs(
    ["💬 Ask", "🔍 Find similar", "👯 Find duplicates", "📋 Browse"]
)

# --- Ask the agent ---------------------------------------------------------
with ask_tab:
    st.subheader("Ask the triage assistant")
    st.caption("It picks the right tool on its own (search, duplicates, priority, etc.).")
    q = st.text_input("Your question", placeholder="e.g. Are there duplicates of TICK-001?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Thinking…"):
            result = run_agent(q, tools, llm_with_tools)
        st.markdown(result["response"])
        if result["tools_used"]:
            st.caption("Tools used: " + ", ".join(result["tools_used"]))

# --- Find similar ----------------------------------------------------------
with similar_tab:
    st.subheader("Find similar open tickets")
    desc = st.text_area("Describe the bug", placeholder="e.g. search shows houses far outside the radius")
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input("How many results", 1, 15, 5)
    with col2:
        pri = st.selectbox("Limit to priority (optional)", ["Any"] + tm.priorities())
    if st.button("Search", key="sim_btn") and desc.strip():
        rows = tm.similar_rows(desc, k=int(k), priority=None if pri == "Any" else pri)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.warning("No similar tickets found.")

# --- Find duplicates -------------------------------------------------------
with dup_tab:
    st.subheader("Find likely duplicates")
    st.caption("Enter a bug description, or an existing ticket ID like TICK-001.")
    dq = st.text_input("Description or ticket ID", key="dup_input")
    if st.button("Check for duplicates", key="dup_btn") and dq.strip():
        res = tm.duplicate_rows(dq, threshold=threshold)
        if res["error"]:
            st.warning(res["error"])
        elif res["rows"]:
            target = res["exclude_id"] or "your description"
            st.success(f"Likely duplicates of {target} (cutoff {threshold:.2f}):")
            st.dataframe(res["rows"], hide_index=True, use_container_width=True)
        else:
            st.info(f"No likely duplicates found at cutoff {threshold:.2f}. Try lowering it in the sidebar.")

# --- Browse ----------------------------------------------------------------
with browse_tab:
    st.subheader("Browse the backlog")
    c1, c2 = st.columns(2)
    with c1:
        bp = st.selectbox("Priority", ["All"] + tm.priorities())
    with c2:
        bc = st.selectbox("Category", ["All"] + tm.categories())
    rows = tm.tickets
    if bp != "All":
        rows = [t for t in rows if t["priority"] == bp]
    if bc != "All":
        rows = [t for t in rows if t["category"] == bc]
    st.write(f"{len(rows)} ticket(s)")
    st.dataframe(
        [{"Ticket": t["ticket_id"], "Title": t["title"],
          "Category": t["category"], "Priority": t["priority"],
          "Created": t["created_date"]} for t in rows],
        hide_index=True, use_container_width=True,
    )

