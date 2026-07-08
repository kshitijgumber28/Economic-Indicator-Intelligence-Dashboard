"""
=====================================================================
  Economic Indicator Intelligence Dashboard — India
  Phase 3: Streamlit Dashboard
  Deploy: Streamlit Community Cloud
=====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="India Economic Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / colour palette ────────────────────────────────────────
C = {
    "india":   "#f97316",
    "us":      "#58a6ff",
    "china":   "#f85149",
    "germany": "#3fb950",
    "uk":      "#a371f7",
    "brazil":  "#ffa657",
    "accent":  "#ff7b72",
    "soft":    "#8b949e",
    "repo":    "#79c0ff",
    "inr":     "#d2a8ff",
    "ip":      "#56d364",
    "bg":      "#0f1117",
    "card":    "#161b22",
    "border":  "#30363d",
}

COUNTRY_COLORS = {
    "India":          C["india"],
    "United States":  C["us"],
    "China":          C["china"],
    "Germany":        C["germany"],
    "United Kingdom": C["uk"],
    "Brazil":         C["brazil"],
}

# ── Global Plotly layout ──────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#161b22", bordercolor=C["border"], borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #c9d1d9;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f97316;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-delta {
    font-size: 0.85rem;
    margin-top: 6px;
}

/* Insight box */
.insight-box {
    background: #161b22;
    border-left: 3px solid #f97316;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0 24px 0;
    color: #c9d1d9;
    font-size: 0.95rem;
    line-height: 1.7;
}
.insight-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #f97316;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Section header */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: white;
    padding-bottom: 6px;
    border-bottom: 1px solid #30363d;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.88rem;
    color: #8b949e;
    margin-bottom: 18px;
}

/* Shock badge */
.shock-badge {
    display: inline-block;
    background: #2a1515;
    border: 1px solid #ff7b72;
    color: #ff7b72;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    fred = pd.read_csv("data/india_fred_all.csv", parse_dates=["date"])
    wb   = pd.read_csv("data/worldbank_all_countries.csv", parse_dates=["date"])
    wb["year"] = wb["date"].dt.year

    inflation = fred[fred["indicator"] == "inflation_yoy_pct"].copy()
    repo      = fred[fred["indicator"] == "repo_rate"].copy()
    usd_inr   = fred[fred["indicator"] == "usd_inr"].copy()
    ind_prod  = fred[fred["indicator"] == "industrial_production"].copy()

    # Normalise IIP to 2015=100
    base = ind_prod[ind_prod["date"].dt.year == 2015]["value"].mean()
    if base and base > 0:
        ind_prod["value"] = (ind_prod["value"] / base) * 100

    # Resample repo to monthly for dual-axis chart
    repo_m = (repo.set_index("date")["value"]
              .resample("MS").ffill().reset_index())
    repo_m["indicator"] = "repo_rate"
    repo_m["country"]   = "India"
    repo_m["source"]    = "FRED"

    return dict(
        fred=fred, wb=wb,
        inflation=inflation, repo=repo, repo_m=repo_m,
        usd_inr=usd_inr, ind_prod=ind_prod,
        gdp_growth     = wb[wb["indicator"] == "gdp_growth_pct"],
        gdp_per_capita = wb[wb["indicator"] == "gdp_per_capita"],
        wb_inflation   = wb[wb["indicator"] == "inflation_pct"],
        unemployment   = wb[wb["indicator"] == "unemployment_pct"],
        current_acct   = wb[wb["indicator"] == "current_account_gdp"],
    )

# ── Shock definitions ─────────────────────────────────────────────
SHOCKS = {
    "Global Financial Crisis (2008)": ("2008-09-01", "2009-06-01",
        "The US housing market collapsed, triggering a global banking crisis. "
        "India's economy was largely insulated because Indian banks had minimal "
        "exposure to the toxic mortgage-backed securities at the centre of the "
        "crisis. However, India felt the impact through trade — exports fell as "
        "Western consumers stopped spending — and the rupee weakened as foreign "
        "investors pulled money out of emerging markets."),

    "Taper Tantrum (2013)": ("2013-05-01", "2013-10-01",
        "In May 2013, US Federal Reserve Chair Ben Bernanke hinted that the Fed "
        "might slow its bond-buying programme. This caused panic in emerging "
        "markets — investors pulled money out of countries like India and rushed "
        "back to the US. The rupee fell sharply from ₹55 to nearly ₹68 per "
        "dollar in a matter of months. The RBI was forced to raise rates to "
        "defend the currency, even though domestic growth was slowing."),

    "Demonetisation (2016)": ("2016-11-01", "2017-03-01",
        "On 8 November 2016, Prime Minister Modi announced that ₹500 and ₹1000 "
        "notes — 86% of all cash in circulation — would no longer be legal tender "
        "overnight. The move was intended to flush out black money and counterfeit "
        "currency. The immediate effect was severe: cash-dependent sectors like "
        "agriculture and informal trade came to a near standstill. Industrial "
        "production and consumption dipped sharply in Q3 FY17."),

    "COVID-19 Pandemic (2020)": ("2020-03-01", "2020-09-01",
        "India imposed one of the world's strictest lockdowns in late March 2020. "
        "Industrial production collapsed by nearly 60% in April 2020 alone — the "
        "steepest single-month fall in modern Indian economic history. GDP "
        "contracted 6.6% for the full year, the worst since Independence. "
        "Counterintuitively, inflation rose rather than fell, because the lockdown "
        "disrupted supply chains (a supply shock), even as demand collapsed. "
        "The RBI cut the repo rate to a historic low of 4% to support growth."),

    "Post-COVID Inflation Surge (2022)": ("2021-10-01", "2022-12-01",
        "As the world reopened after COVID, pent-up demand collided with broken "
        "supply chains, pushing inflation globally. In India, this was amplified "
        "by the Russia-Ukraine war in February 2022, which sent oil and food "
        "prices soaring. India imports 85% of its oil, so energy price shocks "
        "hit hard. Inflation breached the RBI's 6% upper tolerance band, forcing "
        "an aggressive rate-hiking cycle from May 2022."),
}

# ── Insight text library ──────────────────────────────────────────
INSIGHTS = {
    "inflation": (
        "India's inflation has been on a long journey of improvement. In the 2000s, "
        "it averaged around 7% — high and unpredictable, eroding purchasing power "
        "for ordinary households. The turning point came in 2016 when India adopted "
        "a formal inflation targeting framework, giving the RBI a clear mandate: "
        "keep inflation at 4%, with a tolerance band of 2–6%. Since then, inflation "
        "has been noticeably more stable, spending more time within the target band. "
        "The exception was 2022, when global supply shocks pushed it above 6% — "
        "but even then, the RBI responded swiftly with rate hikes, and inflation "
        "returned to target within 18 months."
    ),
    "repo": (
        "The repo rate is the RBI's main lever for controlling the economy. When "
        "inflation is too high, the RBI raises the repo rate — making borrowing "
        "more expensive, which slows spending and cools prices. When growth is "
        "weak, it cuts the rate to encourage borrowing and investment. Looking at "
        "the chart, you can read India's entire monetary policy history: the "
        "aggressive cuts after the 2008 crisis, the long hiking cycle through "
        "2010–12 as inflation stayed stubbornly high, the historic low of 4% "
        "during COVID in 2020, and the sharp hikes in 2022 to fight post-pandemic "
        "inflation."
    ),
    "usd_inr": (
        "The rupee has fallen from around ₹45 to ₹83 per dollar over 25 years — "
        "a depreciation of about 85%. This sounds alarming, but economics explains "
        "most of it. A simple rule called Purchasing Power Parity says that "
        "currencies tend to weaken at roughly the rate of their inflation "
        "advantage over the other country. India's inflation has consistently been "
        "higher than the US by about 3–4 percentage points per year — and the "
        "rupee's depreciation tracks this almost exactly. The rupee hasn't really "
        "become weaker in real terms. The sharper drops you see — in 2008, 2013, "
        "and 2020 — were crisis-driven and temporary."
    ),
    "ind_prod": (
        "The Index of Industrial Production (IIP) measures how much India's "
        "factories, mines, and utilities are producing each month. Think of it "
        "as a real-time pulse check on the economy, since GDP data only comes out "
        "quarterly. The most dramatic feature of this chart is April 2020: "
        "industrial output collapsed nearly 60% in a single month when India's "
        "lockdown shut almost everything down. The recovery was equally sharp — "
        "a classic V-shape driven by pent-up demand. The chart also shows the "
        "gradual upward trend since 2000, reflecting India's industrial expansion, "
        "with a temporary dip around demonetisation in late 2016."
    ),
    "gdp_comparison": (
        "India has been the fastest-growing large economy over the past 25 years, "
        "averaging around 6.5% annual GDP growth — nearly double the global "
        "average. Even during the 2008 Global Financial Crisis, when the US and "
        "Europe contracted, India still grew at 3.9%. The only year India actually "
        "shrank was 2020, during COVID — and even then, it bounced back faster "
        "than almost every other major economy, growing 8.7% in 2021. China has "
        "been India's closest competitor in growth terms, but China's growth rate "
        "has been slowing since 2010 as its economy matures, while India's remains "
        "robust."
    ),
    "gdp_per_capita": (
        "GDP per capita — the average economic output per person — tells a more "
        "human story than total GDP. India's GDP per capita has grown over 5x "
        "since 2000, from around $450 to over $2,400 per person. That represents "
        "hundreds of millions of people being lifted out of poverty within a single "
        "generation. Yet the chart also shows how far India still has to go — the "
        "US GDP per capita is roughly 25x India's. This gap is both a challenge "
        "and an opportunity: it means India has decades of catch-up growth ahead, "
        "powered by a young population, rapid urbanisation, and a booming digital "
        "economy."
    ),
}

def insight(text):
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-label">💡 What this means</div>'
        f'{text}'
        f'</div>',
        unsafe_allow_html=True
    )

def section(title, subtitle=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def add_shocks_plotly(fig, df, row=1, col=1):
    """Add shock shading to a Plotly figure."""
    for name, (start, end, _) in SHOCKS.items():
        short_name = name.split("(")[0].strip()
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=C["accent"], opacity=0.08,
            layer="below", line_width=0,
            row=row, col=col,
        )
    return fig


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🇮🇳 India Economic Dashboard")
    st.markdown("<small style='color:#8b949e'>2000 – 2025 | FRED + World Bank</small>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📈 Inflation & Rates",
         "🌍 India vs World",
         "⚡ Shock Analysis",
         "💱 Rupee Story",
         "📊 GDP & Growth"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("<small style='color:#8b949e'>**Data sources**<br>FRED (St. Louis Fed)<br>World Bank Open Data</small>",
                unsafe_allow_html=True)
    st.markdown("<small style='color:#8b949e'>**Coverage**<br>Jan 2000 – Jan 2025</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════

data = load_data()
inflation  = data["inflation"]
repo       = data["repo"]
repo_m     = data["repo_m"]
usd_inr    = data["usd_inr"]
ind_prod   = data["ind_prod"]
gdp_growth = data["gdp_growth"]
gdp_pc     = data["gdp_per_capita"]
wb_inf     = data["wb_inflation"]
unemploy   = data["unemployment"]
curr_acct  = data["current_acct"]


# ══════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════

if page == "🏠 Overview":

    st.markdown("# 🇮🇳 India Economic Intelligence Dashboard")
    st.markdown(
        "<p style='color:#8b949e; font-size:1rem;'>"
        "25 years of India's macroeconomic story — explained in plain English."
        "</p>", unsafe_allow_html=True
    )
    st.divider()

    # ── KPI cards ─────────────────────────────────────────────────
    india_gdp_avg   = gdp_growth[gdp_growth["country"]=="India"]["value"].mean()
    latest_inf      = inflation.sort_values("date").iloc[-1]["value"]
    latest_repo     = repo.sort_values("date").iloc[-1]["value"]
    latest_inr      = usd_inr.sort_values("date").iloc[-1]["value"]
    india_pc_now    = gdp_pc[gdp_pc["country"]=="India"].sort_values("year").iloc[-1]["value"]
    india_pc_2000   = gdp_pc[(gdp_pc["country"]=="India") &
                              (gdp_pc["year"]==2000)]["value"].values[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, f"{india_gdp_avg:.1f}%",    "Avg GDP Growth",       "2000–2024"),
        (col2, f"{latest_inf:.1f}%",        "Latest Inflation",     "YoY CPI"),
        (col3, f"{latest_repo:.1f}%",       "RBI Repo Rate",        "Current"),
        (col4, f"₹{latest_inr:.1f}",        "USD/INR Rate",         "Latest monthly avg"),
        (col5, f"${india_pc_now:,.0f}",     "GDP per Capita",       "Constant 2015 USD"),
    ]
    for col, val, label, sub in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-delta" style="color:#8b949e">{sub}</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Four-panel overview chart ──────────────────────────────────
    section("The Full Picture",
            "Four key indicators that define India's economic health — 2000 to 2025")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["CPI Inflation (YoY %)", "RBI Repo Rate (%)",
                        "USD/INR Exchange Rate", "Industrial Production (2015=100)"],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # Inflation
    fig.add_trace(go.Scatter(
        x=inflation["date"], y=inflation["value"],
        name="Inflation", line=dict(color=C["india"], width=2),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
    ), row=1, col=1)
    fig.add_hline(y=4, line_dash="dot", line_color=C["soft"],
                  line_width=1, row=1, col=1)
    fig.add_hline(y=6, line_dash="dot", line_color=C["accent"],
                  line_width=1, opacity=0.6, row=1, col=1)

    # Repo
    fig.add_trace(go.Scatter(
        x=repo["date"], y=repo["value"],
        name="Repo Rate", line=dict(color=C["repo"], width=2.5, shape="hv"),
        fill="tozeroy", fillcolor="rgba(121,192,255,0.08)",
    ), row=1, col=2)

    # USD/INR
    fig.add_trace(go.Scatter(
        x=usd_inr["date"], y=usd_inr["value"],
        name="USD/INR", line=dict(color=C["inr"], width=1.8),
    ), row=2, col=1)

    # IIP
    ip_sorted = ind_prod.sort_values("date")
    rolling   = ip_sorted["value"].rolling(12).mean()
    fig.add_trace(go.Scatter(
        x=ip_sorted["date"], y=ip_sorted["value"],
        name="IIP", line=dict(color=C["ip"], width=1.2),
        opacity=0.6,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ip_sorted["date"], y=rolling,
        name="12-month trend", line=dict(color="white", width=2),
        opacity=0.7,
    ), row=2, col=2)

    # Add shock shading to all panels
    for r, c in [(1,1),(1,2),(2,1),(2,2)]:
        for _, (start, end, _) in SHOCKS.items():
            fig.add_vrect(x0=start, x1=end,
                          fillcolor=C["accent"], opacity=0.07,
                          layer="below", line_width=0, row=r, col=c)

    fig.update_layout(**PLOT_LAYOUT, height=580,
                      title_text="", showlegend=False)
    fig.update_annotations(font_color="#c9d1d9")
    st.plotly_chart(fig, use_container_width=True)

    insight(
        "India's macroeconomic story from 2000 to 2025 is one of remarkable growth "
        "interrupted by three distinct shocks — the 2008 Global Financial Crisis, "
        "COVID-19 in 2020, and the post-pandemic inflation surge in 2022. The shaded "
        "red bands mark these periods. Inflation has become progressively more stable "
        "since 2016 when India adopted formal inflation targeting. The rupee has "
        "gradually weakened against the dollar — mostly explained by India's higher "
        "inflation rate, not economic weakness. Industrial production shows a sharp "
        "V-shaped collapse and recovery around COVID."
    )

    # ── Shock timeline ─────────────────────────────────────────────
    st.divider()
    section("Major Economic Shocks",
            "Click any shock to understand what happened and why")

    for shock_name, (start, end, explanation) in SHOCKS.items():
        with st.expander(f"🔴 {shock_name}  ({start[:7]} → {end[:7]})"):
            st.markdown(
                f'<div style="color:#c9d1d9; line-height:1.7;">{explanation}</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════
#  PAGE: INFLATION & RATES
# ══════════════════════════════════════════════════════════════════

elif page == "📈 Inflation & Rates":

    section("Inflation & RBI Repo Rate",
            "How India's central bank fights inflation — the push and pull")

    show_shocks = st.toggle("Show shock periods", value=True)

    # ── Dual axis: inflation + repo ────────────────────────────────
    inf_m = (inflation.set_index("date")["value"]
             .resample("MS").mean().reset_index())
    combined = pd.merge(
        inf_m.rename(columns={"value":"inflation"}),
        repo_m[["date","value"]].rename(columns={"value":"repo"}),
        on="date", how="inner"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=combined["date"], y=combined["inflation"],
        name="Inflation (YoY %)",
        line=dict(color=C["india"], width=2.2),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=combined["date"], y=combined["repo"],
        name="Repo Rate (%)",
        line=dict(color=C["repo"], width=2.5, shape="hv"),
    ), secondary_y=True)

    # RBI target lines
    fig.add_hline(y=4, line_dash="dot", line_color=C["soft"],
                  line_width=1, opacity=0.7, secondary_y=False)
    fig.add_hline(y=6, line_dash="dot", line_color=C["accent"],
                  line_width=1, opacity=0.5, secondary_y=False)

    # Key RBI decision annotations
    annotations = [
        ("2008-10-15", 4.4,  "RBI cuts aggressively\npost-GFC", True),
        ("2010-03-01", 5.25, "Hiking cycle begins\n(inflation too high)", True),
        ("2020-04-15", 4.0,  "Historic cut:\n4% (COVID)", True),
        ("2022-05-04", 4.4,  "Emergency hike\n(inflation >6%)", True),
    ]
    for date, y, label, _ in annotations:
        fig.add_annotation(
            x=date, y=y, text=label,
            ax=0, ay=-45,
            font=dict(size=9, color="white"),
            bgcolor="#161b22", bordercolor=C["soft"],
            borderwidth=1, borderpad=4,
            arrowcolor=C["soft"], arrowwidth=1,
            secondary_y=True,
        )

    if show_shocks:
        for _, (start, end, _) in SHOCKS.items():
            fig.add_vrect(x0=start, x1=end,
                          fillcolor=C["accent"], opacity=0.08,
                          layer="below", line_width=0)

    fig.update_layout(**PLOT_LAYOUT, height=480,
                      title_text="Inflation vs RBI Repo Rate (2001–2024)")
    fig.update_yaxes(title_text="Inflation (%)",
                     color=C["india"], secondary_y=False)
    fig.update_yaxes(title_text="Repo Rate (%)",
                     color=C["repo"], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    insight(INSIGHTS["repo"])

    st.divider()

    # ── Inflation distribution ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section("Inflation by Era",
                "How much has inflation improved since formal targeting?")

        pre  = inflation[inflation["date"] < "2016-01-01"]["value"]
        post = inflation[inflation["date"] >= "2016-01-01"]["value"]

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=pre, name="Pre-2016 (avg {:.1f}%)".format(pre.mean()),
            marker_color=C["accent"], opacity=0.7, nbinsx=25,
        ))
        fig2.add_trace(go.Histogram(
            x=post, name="Post-2016 (avg {:.1f}%)".format(post.mean()),
            marker_color=C["india"], opacity=0.7, nbinsx=20,
        ))
        fig2.add_vline(x=4, line_dash="dot", line_color="white",
                       line_width=1.5, opacity=0.6)
        fig2.add_vline(x=6, line_dash="dot", line_color=C["accent"],
                       line_width=1.5, opacity=0.6)
        fig2.update_layout(**PLOT_LAYOUT, height=340,
                           barmode="overlay",
                           title_text="Distribution of Monthly Inflation Readings")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("Time above RBI Target Band",
                "How often has inflation breached the 2–6% tolerance?")

        total   = len(inflation)
        in_band = len(inflation[inflation["value"].between(2, 6)])
        above   = len(inflation[inflation["value"] > 6])
        below   = len(inflation[inflation["value"] < 2])

        fig3 = go.Figure(go.Pie(
            labels=["Within 2–6% band", "Above 6% (too hot)", "Below 2% (too cold)"],
            values=[in_band, above, below],
            marker_colors=[C["ip"], C["accent"], C["repo"]],
            hole=0.55,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=340,
                           title_text="% of Months in Each Inflation Zone (2001–2025)",
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    insight(INSIGHTS["inflation"])


# ══════════════════════════════════════════════════════════════════
#  PAGE: INDIA VS WORLD
# ══════════════════════════════════════════════════════════════════

elif page == "🌍 India vs World":

    section("India vs The World",
            "Comparing India's performance against 5 major economies")

    indicator_choice = st.selectbox(
        "Select indicator",
        ["GDP Growth (%)", "Inflation (%)", "Unemployment (%)", "Current Account (% GDP)"],
    )

    countries_choice = st.multiselect(
        "Select countries",
        list(COUNTRY_COLORS.keys()),
        default=list(COUNTRY_COLORS.keys()),
    )

    indicator_map = {
        "GDP Growth (%)":          (gdp_growth,  "gdp_growth_pct",     "Annual GDP Growth (%)"),
        "Inflation (%)":           (wb_inf,      "inflation_pct",      "Annual Inflation (%)"),
        "Unemployment (%)":        (unemploy,    "unemployment_pct",   "Unemployment (% of Labour Force)"),
        "Current Account (% GDP)": (curr_acct,   "current_account_gdp","Current Account Balance (% of GDP)"),
    }

    df_plot, ind_key, ylabel = indicator_map[indicator_choice]

    fig = go.Figure()
    for country in countries_choice:
        sub = df_plot[df_plot["country"] == country].sort_values("year")
        if sub.empty:
            continue
        lw    = 3.5 if country == "India" else 1.8
        color = COUNTRY_COLORS.get(country, "#ffffff")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["value"],
            name=country,
            line=dict(color=color, width=lw),
            mode="lines+markers",
            marker=dict(size=5 if country=="India" else 3),
            opacity=1.0 if country=="India" else 0.75,
        ))

    # Shock year lines
    for yr, label in [(2008,"GFC"),(2020,"COVID")]:
        fig.add_vline(x=yr, line_dash="dash",
                      line_color=C["accent"], line_width=1, opacity=0.5)
        fig.add_annotation(x=yr+0.2, y=0.02, text=label,
                           font=dict(size=9, color=C["accent"]),
                           showarrow=False, yref="paper")

    fig.add_hline(y=0, line_color="white", line_width=0.6, opacity=0.3)

    fig.update_layout(**PLOT_LAYOUT, height=480,
                      title_text=f"{indicator_choice} — India vs Major Economies",
                      yaxis_title=ylabel)
    st.plotly_chart(fig, use_container_width=True)

    insight(INSIGHTS["gdp_comparison"])

    # ── Bar chart: latest values ───────────────────────────────────
    st.divider()
    section("Latest Snapshot", "Most recent available value for all countries")

    latest_vals = []
    for country in list(COUNTRY_COLORS.keys()):
        sub = df_plot[df_plot["country"]==country].sort_values("year")
        if not sub.empty:
            last = sub.iloc[-1]
            latest_vals.append({"Country": country,
                                 "Value": round(last["value"],2),
                                 "Year": int(last["year"])})

    latest_df = pd.DataFrame(latest_vals).sort_values("Value", ascending=True)
    bar_colors = [COUNTRY_COLORS.get(c, "#fff") for c in latest_df["Country"]]

    fig2 = go.Figure(go.Bar(
        x=latest_df["Value"],
        y=latest_df["Country"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in latest_df["Value"]],
        textposition="outside",
    ))
    fig2.update_layout(**PLOT_LAYOUT, height=320,
                       title_text=f"Latest {indicator_choice} by Country",
                       xaxis_title=ylabel,
                       showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: SHOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════

elif page == "⚡ Shock Analysis":

    section("Economic Shock Analysis",
            "Deep-diving into India's two biggest crises")

    shock_choice = st.selectbox(
        "Select a shock to analyse",
        list(SHOCKS.keys())
    )

    start, end, explanation = SHOCKS[shock_choice]

    # Window: 18 months before and after
    window_start = str(pd.Timestamp(start) - pd.DateOffset(months=18))[:10]
    window_end   = str(pd.Timestamp(end)   + pd.DateOffset(months=18))[:10]

    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-label">📌 What happened</div>'
        f'{explanation}'
        f'</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # ── Inflation during shock ─────────────────────────────────────
    with col1:
        inf_w = inflation[(inflation["date"] >= window_start) &
                          (inflation["date"] <= window_end)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=inf_w["date"], y=inf_w["value"],
            line=dict(color=C["india"], width=2.2),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
            name="Inflation",
        ))
        fig.add_vrect(x0=start, x1=end,
                      fillcolor=C["accent"], opacity=0.15,
                      layer="below", line_width=0)
        fig.add_hline(y=6, line_dash="dot", line_color=C["accent"],
                      line_width=1, opacity=0.6)
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          title_text="Inflation (YoY %)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Repo rate during shock ─────────────────────────────────────
    with col2:
        repo_w = repo[(repo["date"] >= window_start) &
                      (repo["date"] <= window_end)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=repo_w["date"], y=repo_w["value"],
            line=dict(color=C["repo"], width=2.5, shape="hv"),
            fill="tozeroy", fillcolor="rgba(121,192,255,0.08)",
            name="Repo Rate",
        ))
        fig.add_vrect(x0=start, x1=end,
                      fillcolor=C["accent"], opacity=0.15,
                      layer="below", line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          title_text="RBI Repo Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Industrial production during shock ─────────────────────────
    ip_w = ind_prod[(ind_prod["date"] >= window_start) &
                    (ind_prod["date"] <= window_end)].sort_values("date")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ip_w["date"], y=ip_w["value"],
        line=dict(color=C["ip"], width=1.8),
        fill="tozeroy", fillcolor="rgba(86,211,100,0.07)",
        name="Industrial Production",
    ))
    fig2.add_vrect(x0=start, x1=end,
                   fillcolor=C["accent"], opacity=0.15,
                   layer="below", line_width=0)
    fig2.update_layout(**PLOT_LAYOUT, height=300,
                       title_text="Industrial Production Index (2015=100)")
    st.plotly_chart(fig2, use_container_width=True)

    # ── GDP comparison ─────────────────────────────────────────────
    st.divider()
    section("GDP Impact — India vs Peers",
            "How did different economies hold up during this shock?")

    shock_year  = int(start[:4])
    window_yrs  = list(range(shock_year - 2, shock_year + 4))

    fig3 = go.Figure()
    for country, color in COUNTRY_COLORS.items():
        sub = gdp_growth[(gdp_growth["country"]==country) &
                         (gdp_growth["year"].isin(window_yrs))].sort_values("year")
        if sub.empty: continue
        lw = 3 if country=="India" else 1.6
        fig3.add_trace(go.Scatter(
            x=sub["year"], y=sub["value"],
            name=country,
            line=dict(color=color, width=lw),
            mode="lines+markers",
            marker=dict(size=6 if country=="India" else 4),
            opacity=1.0 if country=="India" else 0.7,
        ))

    fig3.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.4)
    fig3.add_vline(x=shock_year, line_dash="dash",
                   line_color=C["accent"], line_width=1.5, opacity=0.7)
    fig3.update_layout(**PLOT_LAYOUT, height=380,
                       title_text="GDP Growth (%) Around the Shock",
                       yaxis_title="Annual GDP Growth (%)")
    fig3.update_xaxes(tickvals=window_yrs)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: RUPEE STORY
# ══════════════════════════════════════════════════════════════════

elif page == "💱 Rupee Story":

    section("The Rupee Story",
            "Why has the rupee fallen, and should India be worried?")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=usd_inr["date"], y=usd_inr["value"],
            name="₹ per $1 USD",
            line=dict(color=C["inr"], width=2),
            fill="tozeroy", fillcolor="rgba(210,168,255,0.07)",
        ))
        for name, (start, end, _) in SHOCKS.items():
            short = name.split("(")[0].strip()
            fig.add_vrect(x0=start, x1=end,
                          fillcolor=C["accent"], opacity=0.1,
                          layer="below", line_width=0)
        fig.update_layout(**PLOT_LAYOUT, height=400,
                          title_text="USD/INR Exchange Rate (2000–2024)",
                          yaxis_title="₹ per $1 USD")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        inr_start = float(usd_inr[usd_inr["date"].dt.year==2000]["value"].mean())
        inr_end   = float(usd_inr[usd_inr["date"].dt.year==2024]["value"].mean())
        total_dep = (inr_end - inr_start) / inr_start * 100

        for val, label, sub in [
            (f"₹{inr_start:.0f}", "Rate in 2000", "per $1 USD"),
            (f"₹{inr_end:.0f}",   "Rate in 2024", "per $1 USD"),
            (f"{total_dep:.0f}%", "Total depreciation", "over 24 years"),
        ]:
            st.markdown(
                f'<div class="metric-card" style="margin-bottom:12px">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-delta" style="color:#8b949e">{sub}</div>'
                f'</div>', unsafe_allow_html=True
            )

    insight(INSIGHTS["usd_inr"])

    # ── Inflation differential ─────────────────────────────────────
    st.divider()
    section("The Economics Behind It",
            "Inflation differential theory: why currencies depreciate")

    us_inf_a  = wb_inf[wb_inf["country"]=="United States"].set_index("year")["value"]
    in_inf_a  = wb_inf[wb_inf["country"]=="India"].set_index("year")["value"]
    inf_diff  = (in_inf_a - us_inf_a).dropna().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=inf_diff["year"],
        y=inf_diff["value"],
        marker_color=[C["accent"] if v>0 else C["ip"] for v in inf_diff["value"]],
        name="Inflation Gap (India − US)",
        opacity=0.85,
    ))
    fig2.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.4)
    fig2.update_layout(**PLOT_LAYOUT, height=340,
                       title_text="Inflation Differential: India minus US (percentage points)",
                       yaxis_title="Percentage points")
    st.plotly_chart(fig2, use_container_width=True)

    avg_diff = inf_diff["value"].mean()
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-label">💡 The PPP Explanation</div>'
        f'Purchasing Power Parity (PPP) theory predicts that a currency weakens at '
        f'roughly the rate of its inflation advantage over the comparison country. '
        f'India\'s inflation has averaged <strong style="color:{C["india"]}">'
        f'{avg_diff:.1f} percentage points</strong> higher than the US per year. '
        f'Over 24 years, that compounds to roughly {avg_diff*24:.0f}% — very close '
        f'to the actual rupee depreciation of {total_dep:.0f}%. In other words, '
        f'the rupee\'s fall is largely a mathematical consequence of India\'s '
        f'higher inflation, not a sign of economic weakness.'
        f'</div>', unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
#  PAGE: GDP & GROWTH
# ══════════════════════════════════════════════════════════════════

elif page == "📊 GDP & Growth":

    section("GDP & Growth Story",
            "How has the average Indian's standard of living changed?")

    # ── GDP per capita over time ───────────────────────────────────
    fig = go.Figure()
    for country, color in COUNTRY_COLORS.items():
        sub = gdp_pc[gdp_pc["country"]==country].sort_values("year")
        if sub.empty: continue
        lw    = 3 if country=="India" else 1.6
        alpha = 1.0 if country=="India" else 0.65
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["value"],
            name=country,
            line=dict(color=color, width=lw),
            opacity=alpha,
            mode="lines",
        ))

    for yr in [2008, 2020]:
        fig.add_vline(x=yr, line_dash="dash",
                      line_color=C["accent"], line_width=1, opacity=0.5)

    fig.update_layout(**PLOT_LAYOUT, height=450,
                      title_text="GDP per Capita — Constant 2015 USD",
                      yaxis_title="USD per person",
                      yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    insight(INSIGHTS["gdp_per_capita"])

    st.divider()

    # ── Growth rate heatmap ────────────────────────────────────────
    section("GDP Growth Heatmap",
            "Which years were good and bad for each country?")

    countries_order = ["India","China","United States","Germany","United Kingdom","Brazil"]
    pivot = (gdp_growth[gdp_growth["country"].isin(countries_order)]
             .pivot(index="country", columns="year", values="value")
             .reindex(countries_order))

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#c62828"],
            [0.35, "#ef5350"],
            [0.5,  "#161b22"],
            [0.65, "#66bb6a"],
            [1.0,  "#1b5e20"],
        ],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont_size=9,
        colorbar=dict(
          title=dict(text="Growth %", font=dict(color="#c9d1d9")),
          tickfont=dict(color="#c9d1d9")
        ),
    ))
    fig2.update_layout(**PLOT_LAYOUT, height=320,
                       title_text="Annual GDP Growth (%) — Heatmap",
                       xaxis=dict(side="bottom"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-label">💡 Reading the heatmap</div>'
        f'Green = growth, Red = contraction. India\'s row is almost entirely green — '
        f'the only red years are 2020 (COVID). China shows a similar pattern but '
        f'with a clear fading of growth intensity over time (darker green early, '
        f'lighter green recent). The US, Germany, and UK all turned red in both '
        f'2009 (GFC) and 2020 (COVID). Brazil stands out for its instability — '
        f'alternating between strong growth and contractions, reflecting its '
        f'vulnerability to commodity price swings and domestic political cycles.'
        f'</div>', unsafe_allow_html=True
    )
