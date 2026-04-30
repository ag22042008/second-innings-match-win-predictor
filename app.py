# ╔══════════════════════════════════════════════════════════════╗
# ║        IPL WIN PREDICTOR  –  app.py                        ║
# ║  Run:  streamlit run app.py                                 ║
# ╚══════════════════════════════════════════════════════════════╝
#
# What this file does (step by step):
#  1. Loads / trains the ML model from your CSV data
#  2. Shows a nice web page where you fill match info
#  3. Predicts win probability using the model
#  4. Shows charts: win gauge, team stats, head-to-head
#
# Libraries needed:
#   pip install streamlit pandas scikit-learn plotly

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import zipfile
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS  – makes the page look beautiful
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- overall page background ---------- */
.main { background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 100%); }

/* ---------- big title ---------- */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f5a623, #e74c3c, #9b59b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #8899aa;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ---------- section card ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f5a623;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ---------- win probability boxes ---------- */
.prob-box {
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    font-size: 2.4rem;
    font-weight: 900;
}
.prob-batting  { background: linear-gradient(135deg, #1a6b3a, #27ae60); color: #eafaf1; }
.prob-bowling  { background: linear-gradient(135deg, #7b1c1c, #e74c3c); color: #fdf2f2; }
.prob-team-name { font-size: 0.95rem; font-weight: 600; opacity: 0.85; margin-bottom: 0.4rem; }

/* ---------- input labels ---------- */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #ccd6f6 !important;
    font-weight: 600;
}

/* ---------- metric cards ---------- */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ---------- Predict button ---------- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f5a623, #e74c3c);
    color: white;
    font-weight: 800;
    font-size: 1.15rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem;
    transition: transform 0.15s;
}
div.stButton > button:hover { transform: scale(1.02); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101420, #1a1f2e);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Team colors (for charts)
TEAM_COLORS = {
    "Mumbai Indians":               "#004ba0",
    "Chennai Super Kings":          "#f5a623",
    "Kolkata Knight Riders":        "#6a0dad",
    "Royal Challengers Bangalore":  "#cc0000",
    "Sunrisers Hyderabad":          "#ff6600",
    "Delhi Capitals":               "#0078c8",
    "Kings XI Punjab":              "#aa1f2e",
    "Rajasthan Royals":             "#2d4fa1",
}

TEAMS = sorted(TEAM_COLORS.keys())

CITIES = sorted([
    'Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi',
    'Chandigarh','Jaipur','Chennai','Cape Town','Port Elizabeth',
    'Durban','Centurion','East London','Johannesburg','Kimberley',
    'Bloemfontein','Ahmedabad','Cuttack','Nagpur','Dharamsala',
    'Visakhapatnam','Pune','Raipur','Ranchi','Abu Dhabi',
    'Sharjah','Mohali','Bengaluru',
])


# ─────────────────────────────────────────────────────────────────
#  LOAD / TRAIN MODEL  (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_or_train_model():
    """
    Try to load pipe.pkl from disk.
    If that fails (version mismatch etc.), rebuild the model from
    the CSV files that sit next to this script.
    Returns: (pipeline, match_df_clean, delivery_df_features)
    """
    # ── locate data files ──────────────────────────────────────────
    base = os.path.dirname(os.path.abspath(__file__))

    # Support both: data in same folder OR uploaded paths
    candidates_match = [
        os.path.join(base, "matches.csv"),
        os.path.join(base, "matches__1_.csv"),
        "/mnt/user-data/uploads/matches__1_.csv",
    ]
    candidates_del = [
        os.path.join(base, "deliveries.csv"),
        os.path.join(base, "deliveries_csv__1_.zip"),
        "/mnt/user-data/uploads/deliveries_csv__1_.zip",
    ]
    candidates_pkl = [
        os.path.join(base, "pipe.pkl"),
        "/mnt/user-data/uploads/pipe.pkl",
    ]

    match_path = next((p for p in candidates_match if os.path.exists(p)), None)
    del_path   = next((p for p in candidates_del   if os.path.exists(p)), None)

    if match_path is None or del_path is None:
        st.error("❌ Could not find matches.csv / deliveries.csv. "
                 "Put them in the same folder as app.py.")
        st.stop()

    # ── load CSVs ─────────────────────────────────────────────────
    match = pd.read_csv(match_path)

    if del_path.endswith(".zip"):
        with zipfile.ZipFile(del_path) as z:
            with z.open(z.namelist()[0]) as f:
                deliveries = pd.read_csv(f)
    else:
        deliveries = pd.read_csv(del_path)

    # ── clean team names ──────────────────────────────────────────
    renames = {"Delhi Daredevils": "Delhi Capitals",
               "Deccan Chargers":  "Sunrisers Hyderabad"}
    for col in ["team1", "team2", "winner"]:
        for old, new in renames.items():
            match[col] = match[col].str.replace(old, new, regex=False)

    match = match[match["team1"].isin(TEAMS) & match["team2"].isin(TEAMS)]
    match = match[match["dl_applied"] == 0]
    match_clean = match.copy()          # keep for stats

    # ── feature engineering ───────────────────────────────────────
    total_1st = (deliveries[deliveries["inning"] == 1]
                 .groupby("match_id")["total_runs"].sum().reset_index())

    match_df = match[["id","city","winner"]].merge(
        total_1st, left_on="id", right_on="match_id"
    )
    match_df = match_df[["match_id","city","total_runs","winner"]]

    del_df = match_df.merge(deliveries, on="match_id")
    del_df = del_df[del_df["inning"] == 2].copy()

    del_df["total_runs_y"] = pd.to_numeric(del_df["total_runs_y"], errors="coerce")
    del_df["current_score"] = del_df.groupby("match_id")["total_runs_y"].cumsum()
    del_df["runs_left"]     = (del_df["total_runs_x"] + 1) - del_df["current_score"]
    del_df["BallsLeft"]     = 120 - ((del_df["over"] - 1) * 6 + del_df["ball"])

    del_df["player_dismissed"] = (
        del_df["player_dismissed"].fillna("0")
        .apply(lambda x: "0" if x == "0" else "1").astype(int)
    )
    del_df["wickets_left"] = (
        10 - del_df.groupby("match_id")["player_dismissed"].cumsum().values
    )
    del_df["crr"] = (del_df["current_score"] * 6) / (120 - del_df["BallsLeft"])
    del_df["rrr"] = (del_df["runs_left"] * 6) / del_df["BallsLeft"]
    del_df["result"] = (del_df["batting_team"] == del_df["winner"]).astype(int)

    del_df = del_df[(del_df["BallsLeft"] > 0) & (del_df["runs_left"] > 0)]

    final_df = del_df[["city","batting_team","bowling_team","wickets_left",
                        "total_runs_x","runs_left","BallsLeft","crr","rrr","result"]].dropna()
    final_df = final_df[final_df["BallsLeft"] != 0].sample(frac=1, random_state=42)

    X = final_df.iloc[:, :-1]
    y = final_df.iloc[:, -1]

    # ── try loading existing pickle first ─────────────────────────
    pipe = None
    for pkl in candidates_pkl:
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    pipe = pickle.load(f)
                break
            except Exception:
                pipe = None   # version mismatch → retrain

    # ── train if needed ───────────────────────────────────────────
    if pipe is None:
        trf = ColumnTransformer([
            ("ohe", OneHotEncoder(sparse_output=False, drop="first"),
             ["batting_team", "bowling_team", "city"])
        ], remainder="passthrough")
        pipe = Pipeline([("trf", trf),
                         ("model", LogisticRegression(solver="liblinear", max_iter=1000))])
        pipe.fit(X, y)

    return pipe, match_clean, del_df


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Win Predictor")
    st.markdown("---")
    st.markdown("""
**How to use:**
1. Pick batting & bowling team
2. Choose host city
3. Enter target score
4. Enter current match situation
5. Click **Predict Probability**

---
**Model:** Logistic Regression  
**Accuracy:** ~84%  
**Seasons trained:** IPL 2008–2019
""")
    st.markdown("---")
    page = st.radio("Navigate", ["🎯 Predictor", "📊 Team Stats", "⚔️ Head to Head"])


# ─────────────────────────────────────────────────────────────────
#  LOAD DATA + MODEL
# ─────────────────────────────────────────────────────────────────
pipe, match_clean, delivery_features = load_or_train_model()


# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def make_gauge(win_pct: float, team_name: str, color: str) -> go.Figure:
    """Create a speedometer-style gauge for win probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_pct,
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        title={"text": team_name, "font": {"size": 16, "color": "#ccd6f6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#ccd6f6"},
            "bar":  {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  33], "color": "rgba(231,76,60,0.18)"},
                {"range": [33, 66], "color": "rgba(245,166,35,0.18)"},
                {"range": [66,100], "color": "rgba(39,174,96,0.18)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": win_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=260,
        margin=dict(t=30, b=10, l=20, r=20),
        font_color="white",
    )
    return fig


def win_probability_chart(batting_win_pct: float, bat_team: str, bowl_team: str):
    """Horizontal stacked bar showing win split."""
    bat_color  = TEAM_COLORS.get(bat_team,  "#27ae60")
    bowl_color = TEAM_COLORS.get(bowl_team, "#e74c3c")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=bat_team, x=[batting_win_pct], y=[""],
        orientation="h", marker_color=bat_color,
        text=f"  {bat_team}  {batting_win_pct:.1f}%",
        textposition="inside", insidetextanchor="start",
        textfont=dict(size=14, color="white"),
    ))
    fig.add_trace(go.Bar(
        name=bowl_team, x=[100 - batting_win_pct], y=[""],
        orientation="h", marker_color=bowl_color,
        text=f"{bowl_team}  {100-batting_win_pct:.1f}%  ",
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        barmode="stack", height=90,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
#  PAGE 1 — PREDICTOR
# ─────────────────────────────────────────────────────────────────
if "🎯 Predictor" in page:

    st.markdown('<div class="hero-title">🏏 IPL Win Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live second-innings win probability using Machine Learning</div>',
                unsafe_allow_html=True)

    # ── Input section ─────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🏟️ Match Setup</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox("🏏 Batting Team", TEAMS, index=TEAMS.index("Mumbai Indians"))
    with col2:
        bowling_teams = [t for t in TEAMS if t != batting_team]
        bowling_team  = st.selectbox("🎯 Bowling Team", bowling_teams, index=0)
    with col3:
        selected_city = st.selectbox("📍 Host City", CITIES, index=CITIES.index("Mumbai"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📋 Match Situation</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target   = st.number_input("🎯 Target (1st innings total + 1)", min_value=1, max_value=300, value=170)
    with c2:
        score    = st.number_input("📈 Current Score", min_value=0, max_value=300, value=75)
    with c3:
        overs    = st.number_input("⏱️ Overs Completed", min_value=0.1, max_value=19.9,
                                   value=10.0, step=0.1, format="%.1f")
    with c4:
        wickets  = st.number_input("☠️ Wickets Fallen", min_value=0, max_value=9, value=2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Derived stats preview ─────────────────────────────────────
    runs_left  = target - score
    balls_left = 120 - int(overs * 6)
    wkts_left  = 10 - wickets
    crr        = score / overs if overs > 0 else 0
    rrr        = (runs_left * 6) / balls_left if balls_left > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Runs Needed",    str(runs_left))
    m2.metric("Balls Left",     str(balls_left))
    m3.metric("Wickets in Hand",str(wkts_left))
    m4.metric("CRR",  f"{crr:.2f}")
    m5.metric("RRR",  f"{rrr:.2f}")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Win Probability")

    # ── Prediction ────────────────────────────────────────────────
    if predict_btn:
        if batting_team == bowling_team:
            st.error("⚠️ Batting and Bowling teams cannot be the same!")
        elif balls_left <= 0:
            st.error("⚠️ No balls left – the match is over!")
        elif runs_left <= 0:
            st.success(f"🎉 {batting_team} has already won!")
        else:
            input_df = pd.DataFrame({
                "city":         [selected_city],
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "wickets_left": [wkts_left],
                "total_runs_x": [target],
                "runs_left":    [runs_left],
                "BallsLeft":    [balls_left],
                "crr":          [crr],
                "rrr":          [rrr],
            })

            result  = pipe.predict_proba(input_df)[0]
            loss_p  = round(result[0] * 100, 1)
            win_p   = round(result[1] * 100, 1)

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            # Stacked bar
            st.plotly_chart(
                win_probability_chart(win_p, batting_team, bowling_team),
                use_container_width=True, config={"displayModeBar": False},
            )

            # Two gauges
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    make_gauge(win_p, batting_team,
                               TEAM_COLORS.get(batting_team, "#27ae60")),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with g2:
                st.plotly_chart(
                    make_gauge(loss_p, bowling_team,
                               TEAM_COLORS.get(bowling_team, "#e74c3c")),
                    use_container_width=True, config={"displayModeBar": False},
                )

            # Verdict banner
            winner_team = batting_team if win_p >= 50 else bowling_team
            win_val     = win_p        if win_p >= 50 else loss_p
            verdict_color = "#27ae60" if win_p >= 50 else "#e74c3c"
            st.markdown(
                f'<div style="background:{verdict_color}22;border:2px solid {verdict_color};'
                f'border-radius:12px;padding:1rem 1.5rem;text-align:center;margin-top:1rem;">'
                f'<span style="font-size:1.5rem;font-weight:900;color:{verdict_color};">'
                f'{"✅" if win_p >= 50 else "⚠️"}  {winner_team} is FAVOURED  '
                f'({win_val:.1f}% chance to win)</span></div>',
                unsafe_allow_html=True,
            )

            # Input dataframe (for transparency)
            with st.expander("🔍 See input features sent to the model"):
                st.dataframe(input_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  PAGE 2 — TEAM STATS
# ─────────────────────────────────────────────────────────────────
elif "📊 Team Stats" in page:

    st.markdown('<div class="hero-title">📊 Team Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">IPL 2008 – 2019 overall performance</div>',
                unsafe_allow_html=True)

    # ── Compute stats ─────────────────────────────────────────────
    wins         = match_clean.groupby("winner").size().reindex(TEAMS, fill_value=0)
    total_played = (pd.concat([match_clean["team1"], match_clean["team2"]])
                    .value_counts().reindex(TEAMS, fill_value=0))
    losses       = total_played - wins
    win_pct      = (wins / total_played * 100).round(1)

    stats_df = pd.DataFrame({
        "Team":      TEAMS,
        "Played":    total_played.values,
        "Won":       wins.values,
        "Lost":      losses.values,
        "Win %":     win_pct.values,
    }).sort_values("Win %", ascending=False).reset_index(drop=True)

    # ── Win % bar chart ───────────────────────────────────────────
    fig_bar = px.bar(
        stats_df, x="Win %", y="Team", orientation="h",
        color="Team",
        color_discrete_map=TEAM_COLORS,
        title="Win Percentage by Team",
        text="Win %",
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", showlegend=False,
        xaxis=dict(range=[0, 80], gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(categoryorder="total ascending"),
        title_font_size=16,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Wins vs Losses stacked bar ────────────────────────────────
    fig_wl = go.Figure()
    fig_wl.add_trace(go.Bar(
        name="Wins",   x=stats_df["Team"], y=stats_df["Won"],
        marker_color=[TEAM_COLORS[t] for t in stats_df["Team"]],
        text=stats_df["Won"], textposition="auto",
    ))
    fig_wl.add_trace(go.Bar(
        name="Losses", x=stats_df["Team"], y=stats_df["Lost"],
        marker_color="rgba(180,180,180,0.25)",
        text=stats_df["Lost"], textposition="auto",
    ))
    fig_wl.update_layout(
        barmode="group", title="Wins vs Losses",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        title_font_size=16,
    )
    st.plotly_chart(fig_wl, use_container_width=True)

    # ── Wins per season heatmap ───────────────────────────────────
    season_wins = (match_clean.groupby(["Season","winner"])
                   .size().reset_index(name="Wins"))
    season_pivot = (season_wins[season_wins["winner"].isin(TEAMS)]
                    .pivot(index="winner", columns="Season", values="Wins")
                    .fillna(0).astype(int))

    fig_heat = go.Figure(go.Heatmap(
        z=season_pivot.values,
        x=season_pivot.columns.tolist(),
        y=season_pivot.index.tolist(),
        colorscale="YlOrRd",
        text=season_pivot.values,
        texttemplate="%{text}",
        showscale=True,
        hovertemplate="Team: %{y}<br>Season: %{x}<br>Wins: %{z}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Wins Per Season (Heatmap)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=380,
        xaxis=dict(tickangle=-30),
        title_font_size=16,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────
    st.markdown("#### 📋 Full Stats Table")
    st.dataframe(
        stats_df.style
            .background_gradient(subset=["Win %"], cmap="YlGn")
            .format({"Win %": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────
#  PAGE 3 — HEAD TO HEAD
# ─────────────────────────────────────────────────────────────────
elif "⚔️ Head to Head" in page:

    st.markdown('<div class="hero-title">⚔️ Head to Head</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare two teams directly</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A", TEAMS, index=0)
    with col_b:
        team_b = st.selectbox("Team B", [t for t in TEAMS if t != team_a], index=0)

    # ── filter matches between the two ───────────────────────────
    h2h = match_clean[
        ((match_clean["team1"] == team_a) & (match_clean["team2"] == team_b)) |
        ((match_clean["team1"] == team_b) & (match_clean["team2"] == team_a))
    ].copy()

    if h2h.empty:
        st.warning("No matches found between these two teams in the dataset.")
    else:
        wins_a = (h2h["winner"] == team_a).sum()
        wins_b = (h2h["winner"] == team_b).sum()
        total  = len(h2h)

        # Big metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(f"✅ {team_a} Wins", wins_a)
        m2.metric("📋 Total Matches",  total)
        m3.metric(f"✅ {team_b} Wins", wins_b)

        # Pie chart
        fig_pie = go.Figure(go.Pie(
            labels=[team_a, team_b],
            values=[wins_a, wins_b],
            marker_colors=[TEAM_COLORS.get(team_a,"#3498db"),
                           TEAM_COLORS.get(team_b,"#e74c3c")],
            hole=0.45,
            textinfo="label+percent",
            textfont_size=14,
        ))
        fig_pie.update_layout(
            title=f"{team_a} vs {team_b} – Win Split",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
            title_font_size=16,
            height=360,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Season-wise wins
        season_a = (h2h[h2h["winner"] == team_a]
                    .groupby("Season").size().reset_index(name="wins"))
        season_b = (h2h[h2h["winner"] == team_b]
                    .groupby("Season").size().reset_index(name="wins"))

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=season_a["Season"], y=season_a["wins"],
            mode="lines+markers+text", name=team_a,
            line=dict(color=TEAM_COLORS.get(team_a,"#3498db"), width=3),
            marker=dict(size=10),
            text=season_a["wins"], textposition="top center",
        ))
        fig_line.add_trace(go.Scatter(
            x=season_b["Season"], y=season_b["wins"],
            mode="lines+markers+text", name=team_b,
            line=dict(color=TEAM_COLORS.get(team_b,"#e74c3c"), width=3),
            marker=dict(size=10),
            text=season_b["wins"], textposition="top center",
        ))
        fig_line.update_layout(
            title="Season-wise Head-to-Head Wins",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-30),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Wins"),
            title_font_size=16,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Match history table
        st.markdown("#### 📋 Match History")
        h2h_show = h2h[["Season","date","city","team1","team2","winner",
                         "win_by_runs","win_by_wickets"]].copy()
        h2h_show.columns = ["Season","Date","City","Team 1","Team 2",
                             "Winner","Won By Runs","Won By Wickets"]
        st.dataframe(h2h_show.reset_index(drop=True), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#556677;font-size:0.85rem;">'
    "IPL Win Predictor • Logistic Regression Model • Trained on IPL 2008–2019"
    "</p>",
    unsafe_allow_html=True,
)
