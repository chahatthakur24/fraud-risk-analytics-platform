# =============================================================================
# CREDIT CARD FRAUD DETECTION — Streamlit Dashboard
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0f14; }
  [data-testid="stSidebar"]          { background: #13161e; border-right: 1px solid #252a38; }
  [data-testid="stHeader"]           { background: transparent; }
  .metric-card {
    background: #13161e; border: 1px solid #252a38;
    border-radius: 10px; padding: 18px 20px; text-align: center;
  }
  .metric-label { color: #64748b; font-size: 11px; letter-spacing: .12em;
                  text-transform: uppercase; font-family: monospace; }
  .metric-value { color: #e2e8f0; font-size: 28px; font-weight: 700;
                  font-family: monospace; margin: 6px 0 2px; }
  .metric-value.green { color: #00e5a0; }
  .metric-value.red   { color: #ef4444; }
  .metric-value.amber { color: #f59e0b; }
  .verdict-box { border-radius: 12px; padding: 24px 28px; text-align: center; margin-bottom: 16px; }
  .verdict-fraud { background: rgba(239,68,68,0.1); border: 2px solid #ef4444; }
  .verdict-safe  { background: rgba(16,185,129,0.1); border: 2px solid #10b981; }
  .verdict-title { font-size: 22px; font-weight: 700; font-family: monospace; letter-spacing: .06em; }
  .verdict-prob  { font-size: 42px; font-weight: 700; font-family: monospace; }
  .verdict-fraud .verdict-title { color: #ef4444; }
  .verdict-fraud .verdict-prob  { color: #ef4444; }
  .verdict-safe  .verdict-title { color: #10b981; }
  .verdict-safe  .verdict-prob  { color: #10b981; }
  .section-title {
    color: #64748b; font-family: monospace; font-size: 11px;
    letter-spacing: .12em; text-transform: uppercase;
    border-bottom: 1px solid #252a38; padding-bottom: 6px; margin-bottom: 14px;
  }
  .stButton > button {
    background: #00e5a0 !important; color: #000 !important;
    font-family: monospace !important; font-weight: 600 !important;
    border: none !important; border-radius: 6px !important;
    width: 100%; padding: 10px !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")

@st.cache_resource(show_spinner=False)
def load_models():
    models, scalers = {}, {}
    threshold = 0.5
    for key, fname in [("logistic","logistic_regression.pkl"),("rf","random_forest.pkl"),("xgboost","xgboost.pkl")]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p): models[key] = joblib.load(p)
    for key, fname in [("robust","robust_scaler.pkl"),("standard","standard_scaler.pkl")]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p): scalers[key] = joblib.load(p)
    tp = os.path.join(MODEL_DIR, "best_threshold.pkl")
    if os.path.exists(tp): threshold = joblib.load(tp)
    return models, scalers, threshold

models, scalers, THRESHOLD = load_models()
MODELS_LOADED = len(models) > 0

# =============================================================================
# FEATURE ENGINEERING  (mirrors 02_feature_engineering.py)
# =============================================================================
V_COLS       = [f"V{i}" for i in range(1, 29)]
ROBUST_COLS  = ["Amount", "Log_Amount", "Time_Since_First"]
FEATURE_COLS = (V_COLS + ["Hour","Is_Night","Time_Since_First","Log_Amount",
                "High_Amount","Amt_Bucket","V14_x_V17","V14_x_V12",
                "V14_squared","Risk_Score","Risk_Tier","Amount"])

def engineer(time_val, amount, v_vals):
    hour       = int((time_val % 86400) // 3600)
    is_night   = 1 if (hour < 5 or hour > 22) else 0
    log_amount = float(np.log1p(amount))
    high_amount= 1 if amount > 1000 else 0
    bins       = [0, 10, 50, 100, 500, 1000, np.inf]
    amt_bucket = min(max(int(np.searchsorted(bins, amount, side="right") - 1), 0), 5)
    v14, v17, v12 = v_vals[13], v_vals[16], v_vals[11]
    rs = 0
    if amount > 1000:  rs += 30
    elif amount > 500: rs += 15
    if is_night:       rs += 20
    if v14 < -5:       rs += 25
    if v17 < -5:       rs += 15
    if v12 < -5:       rs += 10
    risk_score = min(rs, 100)
    risk_tier  = 0 if risk_score <= 30 else (1 if risk_score <= 60 else 2)
    row = dict(zip(V_COLS, v_vals))
    row.update({"Hour":hour,"Is_Night":is_night,"Time_Since_First":time_val,
                "Log_Amount":log_amount,"High_Amount":high_amount,"Amt_Bucket":amt_bucket,
                "V14_x_V17":v14*v17,"V14_x_V12":v14*v12,"V14_squared":v14**2,
                "Risk_Score":risk_score,"Risk_Tier":risk_tier,"Amount":amount})
    return row

def scale_and_predict(row_dict, model_key="ensemble"):
    df = pd.DataFrame([row_dict])
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    probas = {}
    for key in ["logistic","rf","xgboost"]:
        if models.get(key):
            try:
                probas[key] = float(models[key].predict_proba(df)[0,1])
            except Exception:
                probas[key] = float(models[key].predict_proba(df.values)[0,1])
    if model_key == "ensemble" and probas: prob = float(np.mean(list(probas.values())))
    elif model_key in probas:              prob = probas[model_key]
    else:                                  prob = list(probas.values())[0] if probas else 0.0
    return prob, probas

# =============================================================================
# SESSION STATE
# =============================================================================
if "history" not in st.session_state: st.session_state.history = []
if "stats"   not in st.session_state: st.session_state.stats = {"total":0,"fraud":0,"prob_sum":0.0}

# =============================================================================
# DEMO DATA
# =============================================================================
LEGIT_V = [-1.3598,-0.0728,2.5363,1.3781,-0.3383,0.4624,0.2396,0.0987,0.3638,
            0.0908,-0.5516,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0258,
            0.4039,0.2514,-0.0183,0.2778,-0.1105,0.0669,0.1285,-0.1891,0.1336,-0.0210]
FRAUD_V = [-3.0435,-3.1572,1.0880,2.2886,-1.3547,-1.7196,-2.6826,-0.0744,-1.8737,
           -5.1944,-2.8300,-9.4390,-2.0494,-9.4777,-3.0518,-0.7200,-8.2642,-4.0427,
            0.1305,-0.2302,-0.5296,-0.6310,-0.2925,0.0325,0.6618,-0.3736,-0.1498,-0.0836]

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("*Credit Card Fraud Detection*")
    st.divider()
    page = st.radio("Navigation",
        ["🔍 Score Transaction","📊 Model Analytics","📁 Batch Scoring","📋 Session History"],
        label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
    for name, key in [("Logistic Reg","logistic"),("Random Forest","rf"),("XGBoost","xgboost")]:
        st.markdown(f"{'🟢' if models.get(key) else '🔴'} `{name}`")
    if not MODELS_LOADED:
        st.warning("⚠️ No models found.\nRun:\n```\npython notebooks/03_model_training.py\n```")
    else:
        st.success(f"✅ {len(models)} model(s) loaded")
        st.caption(f"Threshold: `{THRESHOLD:.2f}`")
    st.divider()
    st.caption("Built by [Chahat Thakur](https://github.com/chahatthakur24)")

# =============================================================================
# PAGE: SCORE TRANSACTION
# =============================================================================
if page == "🔍 Score Transaction":
    st.markdown("## 🔍 Score a Transaction")

    stats = st.session_state.stats
    total = stats["total"]
    c1,c2,c3,c4 = st.columns(4)
    for col,label,val,cls in [
        (c1,"Scored",      str(total),                                          ""),
        (c2,"Fraud",       str(stats["fraud"]),                                 "red" if stats["fraud"] else ""),
        (c3,"Fraud Rate",  f"{stats['fraud']/total*100:.1f}%" if total else "—","amber" if stats["fraud"] else "green"),
        (c4,"Avg Score",   f"{stats['prob_sum']/total*100:.1f}%" if total else "—",""),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value {cls}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

        p1,p2,p3,p4 = st.columns(4)
        preset = None
        if p1.button("⚠️ Fraud"):  preset="fraud"
        if p2.button("✅ Legit"):  preset="legit"
        if p3.button("🌙 Night"):  preset="night"
        if p4.button("🎲 Random"): preset="random"

        if preset == "fraud":
            st.session_state.update({"_t":406.0,"_a":1837.20,"_v":FRAUD_V[:]})
        elif preset == "legit":
            st.session_state.update({"_t":406.0,"_a":149.62,"_v":LEGIT_V[:]})
        elif preset == "night":
            st.session_state.update({"_t":82800.0,"_a":520.0,"_v":LEGIT_V[:]})
        elif preset == "random":
            st.session_state.update({"_t":float(np.random.randint(0,172800)),
                                     "_a":round(float(np.random.uniform(1,2000)),2),
                                     "_v":[round(float(np.random.uniform(-4,4)),4) for _ in range(28)]})

        dv = st.session_state.get("_v", LEGIT_V)
        ca,cb,cc = st.columns(3)
        time_val = ca.number_input("Time (seconds)", value=st.session_state.get("_t",406.0), step=1.0, format="%.0f")
        amount   = cb.number_input("Amount (USD)",   value=st.session_state.get("_a",149.62), min_value=0.0, step=0.01)
        model_choice = cc.selectbox("Model", ["ensemble","xgboost","rf","logistic"],
            format_func=lambda x:{"ensemble":"Ensemble (All)","xgboost":"XGBoost","rf":"Random Forest","logistic":"Logistic Reg"}[x])

        st.markdown('<div class="section-title" style="margin-top:14px">PCA Features (V1–V28)</div>', unsafe_allow_html=True)
        v_vals = []
        rows = [st.columns(7) for _ in range(4)]
        for i, col in enumerate([c for row in rows for c in row]):
            if i >= 28: break
            v_vals.append(col.number_input(f"V{i+1}", value=float(dv[i]),
                step=0.01, format="%.4f", key=f"vinp{i+1}"))

        st.markdown("")
        score_btn = st.button("🔍 Score Transaction", use_container_width=True, disabled=not MODELS_LOADED)

    with right:
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        if score_btn:
            with st.spinner("Scoring..."):
                t0 = time.time()
                row  = engineer(time_val, amount, v_vals)
                prob, probas = scale_and_predict(row, model_choice)
                ms = int((time.time()-t0)*1000)

            is_fraud = prob >= THRESHOLD
            pct = prob * 100
            tier_label = {0:"Low",1:"Medium",2:"High"}[int(row["Risk_Tier"])]
            vcls = "fraud" if is_fraud else "safe"
            vtxt = "⚠️ FRAUDULENT" if is_fraud else "✅ LEGITIMATE"

            st.markdown(f"""
            <div class="verdict-box verdict-{vcls}">
              <div class="verdict-title">{vtxt}</div>
              <div class="verdict-prob">{pct:.1f}%</div>
              <div style="color:#64748b;font-size:12px;margin-top:4px">fraud probability</div>
            </div>""", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":28,"color":"#e2e8f0"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#64748b","tickfont":{"color":"#64748b"}},
                       "bar":{"color":"#ef4444" if is_fraud else "#10b981","thickness":0.3},
                       "bgcolor":"#1c2030",
                       "steps":[{"range":[0,30],"color":"rgba(16,185,129,0.1)"},
                                 {"range":[30,60],"color":"rgba(245,158,11,0.1)"},
                                 {"range":[60,100],"color":"rgba(239,68,68,0.1)"}],
                       "threshold":{"line":{"color":"#fff","width":2},
                                    "thickness":0.75,"value":THRESHOLD*100}}))
            fig_g.update_layout(height=180, margin=dict(t=20,b=10,l=20,r=20),
                                 paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_g, use_container_width=True)

            d1,d2 = st.columns(2)
            d1.metric("Risk Tier",    tier_label)
            d2.metric("Risk Score",   f"{int(row['Risk_Score'])} / 100")
            d1.metric("Threshold",    f"{THRESHOLD:.2f}")
            d2.metric("Process Time", f"{ms} ms")

            st.markdown('<div class="section-title" style="margin-top:8px">Key Signals</div>', unsafe_allow_html=True)
            sc = st.columns(2)
            for i,(sig,active) in enumerate({"High Amount":amount>1000,"Night Txn":row["Is_Night"]==1,
                                              "V14 Anomaly":v_vals[13]<-5,"V17 Anomaly":v_vals[16]<-5}.items()):
                sc[i%2].markdown(f"{'🔴' if active else '⚪'} `{sig}`")

            if len(probas) > 1:
                st.markdown('<div class="section-title" style="margin-top:12px">Model Breakdown</div>', unsafe_allow_html=True)
                labels = {"logistic":"Logistic Reg","rf":"Random Forest","xgboost":"XGBoost"}
                fig_b = go.Figure(go.Bar(
                    x=[p*100 for p in probas.values()],
                    y=[labels.get(k,k) for k in probas.keys()],
                    orientation="h",
                    marker_color=["#ef4444" if p>=THRESHOLD else "#10b981" for p in probas.values()],
                    text=[f"{p*100:.1f}%" for p in probas.values()],
                    textposition="outside", textfont={"color":"#e2e8f0","size":11}))
                fig_b.update_layout(
                    height=140, margin=dict(t=5,b=5,l=10,r=40),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0,115],showgrid=False,zeroline=False,
                               tickfont={"color":"#64748b"},ticksuffix="%"),
                    yaxis=dict(showgrid=False,tickfont={"color":"#e2e8f0"}),
                    font_color="#e2e8f0")
                st.plotly_chart(fig_b, use_container_width=True)

            st.session_state.history.insert(0,{"fraud":is_fraud,"prob":prob,
                "amount":amount,"tier":tier_label,"model":model_choice,"time":time.strftime("%H:%M:%S")})
            st.session_state.stats["total"]    += 1
            st.session_state.stats["prob_sum"] += prob
            if is_fraud: st.session_state.stats["fraud"] += 1
        else:
            st.info("☝️ Fill in the details and click **Score Transaction**.")
            if not MODELS_LOADED:
                st.error("Models not loaded — run the training pipeline first.")

# =============================================================================
# PAGE: MODEL ANALYTICS
# =============================================================================
elif page == "📊 Model Analytics":
    st.markdown("## 📊 Model Analytics")

    perf = {"Model":["Logistic Regression","Random Forest","XGBoost","Ensemble"],
            "ROC-AUC":[0.974,0.981,0.983,0.984],"PR-AUC":[0.742,0.863,0.876,0.881],
            "Precision":[0.821,0.877,0.891,0.896],"Recall":[0.831,0.839,0.847,0.849],
            "F1":[0.826,0.858,0.869,0.872]}
    df_perf = pd.DataFrame(perf).set_index("Model")
    st.markdown("### Benchmark Performance")
    st.dataframe(df_perf.style.highlight_max(axis=0,color="#00e5a033").format("{:.3f}"),
                 use_container_width=True)

    st.markdown("### Metric Comparison")
    colors  = ["#3b82f6","#10b981","#f59e0b","#00e5a0"]
    metrics = ["ROC-AUC","PR-AUC","Precision","Recall","F1"]
    fig = go.Figure()
    for i,m in enumerate(perf["Model"]):
        fig.add_trace(go.Bar(name=m, x=metrics, y=[perf[k][i] for k in metrics],
                             marker_color=colors[i],
                             text=[f"{perf[k][i]:.3f}" for k in metrics],
                             textposition="outside", textfont={"size":10,"color":"#e2e8f0"}))
    fig.update_layout(barmode="group", height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,30,0.6)",
        legend=dict(bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0"),
        xaxis=dict(tickfont={"color":"#e2e8f0"},gridcolor="#252a38"),
        yaxis=dict(tickfont={"color":"#e2e8f0"},gridcolor="#252a38",range=[0.65,1.02]),
        font_color="#e2e8f0", margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Feature Importances (XGBoost)")
    feats = ["V14","V17","V12","V10","V3","V4","Amount","V11","V16","V7"]
    imps  = [0.31,0.22,0.14,0.10,0.07,0.06,0.04,0.03,0.02,0.01]
    fig2 = go.Figure(go.Bar(x=imps[::-1], y=feats[::-1], orientation="h",
        marker=dict(color=imps[::-1], colorscale=[[0,"#1c2030"],[1,"#00e5a0"]]),
        text=[f"{v:.2f}" for v in imps[::-1]], textposition="outside",
        textfont={"color":"#e2e8f0","size":11}))
    fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,22,30,0.6)",
        xaxis=dict(showgrid=False,tickfont={"color":"#64748b"}),
        yaxis=dict(showgrid=False,tickfont={"color":"#e2e8f0"}),
        margin=dict(t=10,b=10,l=10,r=50), font_color="#e2e8f0")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Dataset Overview")
    m1,m2,m3,m4 = st.columns(4)
    for col,label,val,cls in [(m1,"Total Transactions","284,807",""),
                               (m2,"Fraudulent","492","red"),
                               (m3,"Fraud Rate","0.17%","amber"),
                               (m4,"Features","31 → 42 engineered","green")]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value {cls}">{val}</div></div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: BATCH SCORING
# =============================================================================
elif page == "📁 Batch Scoring":
    st.markdown("## 📁 Batch Scoring")
    st.markdown("Upload a CSV with columns `Time`, `Amount`, `V1`–`V28`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw):,} rows loaded.** Preview:")
        st.dataframe(df_raw.head(5), use_container_width=True)
        missing = [c for c in ["Time","Amount"]+V_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif not MODELS_LOADED:
            st.error("Models not loaded.")
        else:
            if st.button("🚀 Run Batch Scoring", use_container_width=True):
                results = []
                bar = st.progress(0, text="Scoring...")
                for idx, row in df_raw.iterrows():
                    v = [float(row.get(f"V{j}",0)) for j in range(1,29)]
                    eng = engineer(float(row["Time"]), float(row["Amount"]), v)
                    prob, _ = scale_and_predict(eng, "ensemble")
                    results.append({"Transaction":idx,"Amount":row["Amount"],
                        "Fraud Probability":round(prob,4),
                        "Prediction":"FRAUD" if prob>=THRESHOLD else "LEGIT",
                        "Risk Score":int(eng["Risk_Score"]),
                        "Risk Tier":{0:"Low",1:"Medium",2:"High"}[int(eng["Risk_Tier"])]})
                    bar.progress((idx+1)/len(df_raw))
                df_out = pd.DataFrame(results)
                bar.empty()
                fn = (df_out["Prediction"]=="FRAUD").sum()
                c1,c2,c3 = st.columns(3)
                c1.metric("Total Scored",  len(df_out))
                c2.metric("Fraud Detected", fn)
                c3.metric("Fraud Rate",    f"{fn/len(df_out)*100:.2f}%")
                st.dataframe(df_out.style.applymap(
                    lambda v: "color:#ef4444" if v=="FRAUD" else "color:#10b981",
                    subset=["Prediction"]), use_container_width=True)
                st.download_button("⬇️ Download Results",
                    df_out.to_csv(index=False).encode(),
                    "fraud_predictions.csv","text/csv",use_container_width=True)
    else:
        st.info("👆 Upload a CSV file to begin.")
        sample = pd.DataFrame([[406,149.62]+LEGIT_V[:3]+["..."]],
            columns=["Time","Amount","V1","V2","V3","..."])
        st.dataframe(sample, use_container_width=True)

# =============================================================================
# PAGE: SESSION HISTORY
# =============================================================================
elif page == "📋 Session History":
    st.markdown("## 📋 Session History")
    history = st.session_state.history
    if not history:
        st.info("No transactions scored yet.")
    else:
        df_h = pd.DataFrame(history)
        fn   = int(df_h["fraud"].sum())
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Scored",   len(df_h))
        c2.metric("Fraud Detected", fn)
        c3.metric("Fraud Rate",     f"{fn/len(df_h)*100:.1f}%")

        st.markdown("### Probability Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[h["prob"]*100 for h in reversed(history)],
            mode="lines+markers",
            line=dict(color="#00e5a0",width=2),
            marker=dict(color=["#ef4444" if h["fraud"] else "#10b981" for h in reversed(history)],size=8),
            fill="tozeroy", fillcolor="rgba(0,229,160,0.06)"))
        fig.add_hline(y=THRESHOLD*100, line_dash="dash",
                      line_color="#f59e0b", annotation_text="Threshold")
        fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,22,30,0.6)",
            xaxis=dict(showgrid=False,tickfont={"color":"#64748b"},title="Transaction #"),
            yaxis=dict(gridcolor="#252a38",tickfont={"color":"#e2e8f0"},title="Fraud %",range=[0,105]),
            margin=dict(t=10,b=10), font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### All Transactions")
        df_d = df_h[["time","amount","prob","fraud","tier","model"]].copy()
        df_d.columns = ["Time","Amount","Fraud Prob","Is Fraud","Risk Tier","Model"]
        df_d["Fraud Prob"] = df_d["Fraud Prob"].apply(lambda x: f"{x*100:.1f}%")
        df_d["Is Fraud"]   = df_d["Is Fraud"].apply(lambda x: "⚠️ FRAUD" if x else "✅ LEGIT")
        st.dataframe(df_d, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.stats   = {"total":0,"fraud":0,"prob_sum":0.0}
            st.rerun()
