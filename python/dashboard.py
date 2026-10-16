"""Streamlit dashboard for the Wattlebrook governed data platform.

Read-only: it only runs SELECTs against the reports.* views (plus the silver link
for the person lookup). Run the pipeline first, then:

    pip install streamlit
    streamlit run python/dashboard.py

Connection settings come from .env (same as the pipeline scripts).
"""
import os
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Wattlebrook data platform", layout="wide")


@st.cache_resource
def connect():
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "wattlebrook_db"),
        user=os.getenv("POSTGRES_USER", "wbu_admin"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        autocommit=True,
    )
    conn.read_only = True  # the dashboard can never change data
    return conn


@st.cache_data(ttl=60)
def query(sql, params=None):
    with connect().cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def show(df, **kw):
    st.dataframe(df, hide_index=True, width="stretch", **kw)


try:
    connect()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot connect to the database: {e}")
    st.info("Start it with `docker compose up -d`, run the pipeline, then reload.")
    st.stop()

st.title("Wattlebrook University governed data platform")
st.caption("Synthetic data. Every number below is read live from the reports.* views.")

tab_health, tab_run, tab_cat, tab_id, tab_ret, tab_mask, tab_lookup = st.tabs(
    ["Health checks", "Pipeline run", "Catalogue", "Identity", "Retention", "Masking", "Person lookup"]
)

# ---------------------------------------------------------------- health checks
with tab_health:
    checks = query("SELECT * FROM reports.checks")
    passed = int((checks["result"] == "PASS").sum())
    c1, c2 = st.columns(2)
    c1.metric("Checks passing", f"{passed} / {len(checks)}")
    c2.metric("Failing", len(checks) - passed)

    def colour(v):
        return {"PASS": "color: #1a7f37; font-weight: 600", "FAIL": "color: #cf222e; font-weight: 600"}.get(v, "")

    st.dataframe(checks.style.map(colour, subset=["result"]), hide_index=True, width="stretch")
    if (checks["result"] == "FAIL").any():
        st.warning(
            "A FAIL is real, not a bug in the dashboard. For example the retention check fails if records "
            "passed their use-by date since the last pipeline run. Re-run `python python/run_pipeline.py`."
        )

# ------------------------------------------------------------------ pipeline run
with tab_run:
    runs = query("SELECT * FROM reports.pipeline_runs")
    st.subheader("Steps of the most recent run")
    show(runs)
    if not runs.empty:
        st.bar_chart(runs.set_index("step_name")["seconds"].astype(float), y_label="seconds")
    st.subheader("Failed steps (all time)")
    failed = query("SELECT * FROM reports.failed_steps")
    if failed.empty:
        st.success("No failed steps logged.")
    else:
        show(failed)

# ------------------------------------------------------------------- catalogue
with tab_cat:
    cat = query("SELECT * FROM reports.catalogue")
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(cat))
    c2.metric("With personal data", int((cat["pii_columns"] > 0).sum()))
    c3.metric("Without lineage", int((cat["comes_from"].str.startswith("unknown")).sum()))
    layers = ["All"] + sorted(cat["layer"].unique())
    pick = st.selectbox("Layer", layers)
    show(cat if pick == "All" else cat[cat["layer"] == pick])

# -------------------------------------------------------------------- identity
with tab_id:
    acc = query("SELECT * FROM reports.identity_accuracy").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real people (answer key)", f"{int(acc.real_people):,}")
    c2.metric("People found", f"{int(acc.people_found):,}")
    c3.metric("Precision", f"{acc.precision_pct}%")
    c4.metric("Recall", f"{acc.recall_pct}%")
    st.caption(
        f"{int(acc.correct_links):,} correct links, {int(acc.wrong_links):,} wrong links, "
        f"{int(acc.missed_links):,} missed links."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Which systems each person is in")
        overlap = query("SELECT * FROM reports.identity_overlap")
        st.bar_chart(overlap.set_index("systems")["people"])
        show(overlap)
    with right:
        st.subheader("How each source record was matched")
        rules = query(
            "SELECT match_rule, confidence, count(*) AS link_rows FROM silver.link_person "
            "GROUP BY 1, 2 ORDER BY 3 DESC"
        )
        show(rules)
        st.caption("R1 same email, R2 same surname + date of birth + phone, R3 same name + date of birth.")

# ------------------------------------------------------------------- retention
with tab_ret:
    st.subheader("What retention did on its last run")
    show(query("SELECT * FROM reports.retention_results"))
    st.subheader("Personal data past its use-by date right now")
    overdue = query(
        "SELECT policy_id, count(*) AS records, min(retention_expiry_date) AS oldest "
        "FROM reports.retention_overdue GROUP BY 1 ORDER BY 1"
    )
    if overdue.empty:
        st.success("Nothing overdue.")
    else:
        st.warning("These records expired since the last run. The next pipeline run will clear them.")
        show(overdue)

# --------------------------------------------------------------------- masking
with tab_mask:
    st.caption("Counts of real personal data found in the masked test tables. Every count should be 0.")
    mask = query("SELECT * FROM reports.masking_check")
    show(mask)
    if int(mask["found"].sum()) == 0:
        st.success("No real personal data in the masked test tables.")
    else:
        st.error("Real personal data leaked into the test tables.")

# --------------------------------------------------------------- person lookup
with tab_lookup:
    st.caption("Type any source ID (for example STU0000001, EMP000001 or CRM0000001) to see the person it belongs to.")
    sid = st.text_input("Source record ID", value="STU0000001").strip().upper()
    if sid:
        links = query(
            "SELECT l.person_id, l.system_name, l.source_record_id, l.match_rule, l.confidence "
            "FROM silver.link_person l "
            "WHERE l.person_id IN (SELECT person_id FROM silver.link_person WHERE source_record_id = %s) "
            "ORDER BY l.system_name",
            (sid,),
        )
        if links.empty:
            st.info("No record with that ID.")
        else:
            st.write(f"Person **{links.iloc[0].person_id}** is made of {len(links)} source record(s):")
            show(links)
