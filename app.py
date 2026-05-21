import streamlit as st

st.set_page_config(
    page_title="AAPL Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports (guarded so app survives partial implementations) ─────────

try:
    from data.live_fetcher import fetch_and_update
    _fetcher_ok = True
except ImportError:
    _fetcher_ok = False

try:
    from training.retrain_pipeline import run_retrain
    _retrain_ok = True
except ImportError:
    _retrain_ok = False

try:
    from inference.forecaster import generate_forecast
    _forecaster_ok = True
except ImportError:
    _forecaster_ok = False

try:
    from evaluation.comparator import run_evaluation
    _eval_ok = True
except ImportError:
    _eval_ok = False

# ── Stdlib / third-party ─────────────────────────────────────────────────────

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Path constants ───────────────────────────────────────────────────────────

ROLLING_CSV  = Path("data/aapl_rolling.csv")
FEATURES_CSV = Path("data/aapl_features.csv")
FORECAST_LOG = Path("logs/forecast_log.jsonl")
RETRAIN_LOG  = Path("logs/retrain_log.jsonl")
FETCH_LOG    = Path("logs/fetch_log.jsonl")
EVAL_LOG     = Path("logs/eval_log.jsonl")
MODELS_DIR   = Path("models")

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _read_last_jsonl(path: Path) -> dict | None:
    try:
        lines = path.read_text().strip().split("\n")
        return json.loads(lines[-1]) if lines and lines[-1] else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines if l.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# ── Sidebar — system status ───────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AAPL Forecaster")
    st.caption("BiLSTM + Bahdanau Attention  •  100-day window  •  7-day horizon")
    st.divider()

    st.subheader("System Status")

    # Data freshness
    if ROLLING_CSV.exists():
        df_side = pd.read_csv(ROLLING_CSV, index_col=0, parse_dates=True)
        last_day = df_side.index[-1].date()
        today    = datetime.now(timezone.utc).date()
        days_old = (today - last_day).days
        color    = "🟢" if days_old <= 1 else ("🟡" if days_old <= 5 else "🔴")
        st.write(f"{color} **Data** — {last_day} ({days_old}d ago)")
    else:
        st.write("🔴 **Data** — not fetched")

    # Model freshness
    pt_files = sorted(MODELS_DIR.glob("model_v*.pt"))
    if pt_files:
        model_name = pt_files[-1].stem
        st.write(f"🟢 **Model** — {model_name}")
    else:
        st.write("🔴 **Model** — not trained")

    # Last forecast
    last_fc = _read_last_jsonl(FORECAST_LOG)
    if last_fc:
        fc_ts = last_fc.get("run_timestamp", "")[:10]
        st.write(f"🟢 **Forecast** — {fc_ts}")
    else:
        st.write("🔴 **Forecast** — not generated")

    # Last eval
    last_ev = _read_last_jsonl(EVAL_LOG)
    if last_ev:
        ev_ts = last_ev.get("evaluated_at", "")[:10]
        st.write(f"🟢 **Evaluation** — {ev_ts}")
    else:
        st.write("⚪ **Evaluation** — pending future dates")

    st.divider()
    st.caption(
        "Forecasts show estimated trend direction only. "
        "Not financial advice."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Data",
    "🔮 Forecast",
    "⚙️ Model & Controls",
    "📊 Evaluation",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Data Status
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not ROLLING_CSV.exists():
        st.info("No data loaded yet. Go to **Model & Controls** and click Refresh Data.")
        st.stop()

    df = pd.read_csv(ROLLING_CSV, index_col=0, parse_dates=True)
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    delta  = float(latest["Close"] - prev["Close"])
    pct    = delta / float(prev["Close"]) * 100

    # ── Top metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",    f"${latest['Close']:.2f}",  f"{delta:+.2f}")
    c2.metric("Change",          f"{pct:+.2f}%",
              delta_color="normal" if pct >= 0 else "inverse")
    c3.metric("Last Trading Day", str(df.index[-1].date()))
    c4.metric("Days Loaded",     f"{len(df):,}")

    last_fetch = _read_last_jsonl(FETCH_LOG)
    if last_fetch:
        status = last_fetch.get("status", "unknown")
        icon   = "✅" if status == "success" else "⚠️"
        st.caption(
            f"{icon} Last fetch: {last_fetch.get('timestamp', '—')}  "
            f"| Status: {status}  "
            f"| Window: {last_fetch.get('date_range_start','—')} → "
            f"{last_fetch.get('date_range_end','—')}"
        )

    st.divider()

    # ── 90-day price chart ──
    st.subheader("Recent Price History")
    cutoff_90 = df.index[-1] - pd.Timedelta(days=90)
    recent    = df[df.index >= cutoff_90]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=recent.index, y=recent["Close"],
        mode="lines", name="Close",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 180, 216, 0.07)",
    ))
    fig_hist.update_layout(
        title="AAPL Close Price — Last 90 Trading Days",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=350,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── OHLCV snapshot ──
    st.subheader("Latest 10 Trading Days")
    tail_df = df.tail(10)[["Open","High","Low","Close","Volume"]].copy()
    tail_df.index = tail_df.index.date
    tail_df["Volume"] = tail_df["Volume"].map(lambda v: f"{int(v):,}")
    for col in ["Open","High","Low","Close"]:
        tail_df[col] = tail_df[col].map(lambda v: f"${v:.2f}")
    st.dataframe(tail_df[::-1], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.warning(
        "⚠️ **Estimated trend direction only.** This is not a price target and "
        "does not constitute financial advice. Stock prices are influenced by "
        "earnings surprises, macro events, and news that no price-based model "
        "can observe or predict."
    )

    if not ROLLING_CSV.exists():
        st.info("No data available. Refresh data from the Model & Controls tab.")
        st.stop()

    if not FORECAST_LOG.exists():
        st.info(
            "No forecast generated yet. Go to **Model & Controls**, "
            "retrain the model, then click **Generate Forecast**."
        )
        st.stop()

    df       = pd.read_csv(ROLLING_CSV, index_col=0, parse_dates=True)
    last_rec = _read_last_jsonl(FORECAST_LOG)

    if not last_rec or "forecast" not in last_rec:
        st.error("Forecast log exists but last record is malformed.")
        st.stop()

    fc         = last_rec["forecast"]
    fc_df      = pd.DataFrame(fc)
    fc_df["date"] = pd.to_datetime(fc_df["date"])

    cutoff_60  = df.index[-1] - pd.Timedelta(days=60)
    actual_60  = df[df.index >= cutoff_60]["Close"]
    last_close = float(actual_60.iloc[-1])
    last_date  = actual_60.index[-1]

    # ── Forecast summary cards ──
    first_pred = fc_df["predicted_price"].iloc[0]
    last_pred  = fc_df["predicted_price"].iloc[-1]
    week_delta = last_pred - last_close
    week_pct   = week_delta / last_close * 100
    avg_pred   = fc_df["predicted_price"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price",      f"${last_close:.2f}")
    c2.metric("Tomorrow (est.)",    f"${first_pred:.2f}",
              f"{first_pred - last_close:+.2f}")
    c3.metric("7-Day Trend (est.)", f"${last_pred:.2f}",
              f"{week_pct:+.2f}%")
    c4.metric("Model",
              last_rec.get("model_version", "—")[-8:],
              last_rec.get("run_timestamp", "—")[:10])

    st.divider()

    # ── Main chart ──
    fig = go.Figure()

    # Actual close (60 days)
    fig.add_trace(go.Scatter(
        x=actual_60.index, y=actual_60.values,
        mode="lines", name="Actual Close",
        line=dict(color="#00b4d8", width=2),
    ))

    # Connecting bridge: last actual → first forecast point
    bridge_x = [last_date, fc_df["date"].iloc[0]]
    bridge_y = [last_close, fc_df["predicted_price"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines", name="Bridge",
        line=dict(color="orange", dash="dot", width=1),
        showlegend=False,
    ))

    # Uncertainty band (upper, then lower fills back to upper)
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["upper_bound"],
        fill=None, line=dict(width=0),
        showlegend=False, name="Upper Bound",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["lower_bound"],
        fill="tonexty",
        fillcolor="rgba(255, 165, 0, 0.18)",
        line=dict(width=0),
        showlegend=True, name="Uncertainty Band (±1.5%)",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["predicted_price"],
        mode="lines+markers",
        name="Model Estimated Trend (not guaranteed)",
        line=dict(color="orange", dash="dash", width=2),
        marker=dict(size=7, symbol="circle"),
    ))

    # Forecast-start vertical line
    fig.add_vline(
        x=int(last_date.timestamp() * 1000),
        line_dash="dot", line_color="rgba(180,180,180,0.5)",
        annotation_text="Forecast start",
        annotation_position="top right",
        annotation_font_color="gray",
    )

    fig.update_layout(
        title="AAPL — Last 60 Days + 7-Day Estimated Trend",
        xaxis_title="Date", yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        template="plotly_dark",
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast values table ──
    st.subheader("7-Day Forecast Detail")
    display_fc = fc_df.copy()
    display_fc["date"] = display_fc["date"].dt.strftime("%a %b %d, %Y")
    display_fc["predicted_price"] = display_fc["predicted_price"].map(lambda v: f"${v:.2f}")
    display_fc["lower_bound"]     = display_fc["lower_bound"].map(lambda v: f"${v:.2f}")
    display_fc["upper_bound"]     = display_fc["upper_bound"].map(lambda v: f"${v:.2f}")
    display_fc.columns = ["Date", "Estimated Price", "Lower Bound", "Upper Bound"]
    display_fc.index   = range(1, 8)
    st.dataframe(display_fc, use_container_width=True)

    st.caption(
        f"Forecast generated: {last_rec.get('run_timestamp','—')[:19]} UTC  "
        f"| Based on data through: {last_rec.get('last_actual_date','—')}  "
        f"| {last_rec.get('disclaimer','')}"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model & Controls
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Model Status")

    last_retrain = _read_last_jsonl(RETRAIN_LOG)
    pt_files     = sorted(MODELS_DIR.glob("model_v*.pt"))

    if last_retrain and last_retrain.get("status") == "success":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Version",  last_retrain.get("model_version", "—"))
        c2.metric("Last Retrain",   last_retrain.get("run_date", "—"))
        c3.metric("Best Val Loss",  f"{last_retrain.get('best_val_loss', 0):.6f}")
        c4.metric("Epochs Trained", last_retrain.get("epochs_trained", "—"))

        st.caption(
            f"Early stopped: {last_retrain.get('early_stopped','—')}  "
            f"| Train samples: {last_retrain.get('train_samples','—')}  "
            f"| Val samples: {last_retrain.get('val_samples','—')}  "
            f"| Duration: {last_retrain.get('duration_seconds',0):.1f}s"
        )
    else:
        st.info("No successful retrain on record yet.")

    if pt_files:
        with st.expander("Model artifacts on disk"):
            for f in reversed(pt_files):
                size_mb = f.stat().st_size / 1e6
                st.caption(f"  `{f.name}`  —  {size_mb:.1f} MB")

    st.divider()

    # ── Action buttons ──
    st.subheader("Pipeline Controls")
    st.caption(
        "Run these in order on first setup: Refresh → Retrain → Forecast. "
        "After that, refresh daily and forecast as needed."
    )

    col1, col2, col3, col4 = st.columns(4)

    # 1. Refresh data
    with col1:
        st.markdown("**Step 1**")
        if st.button("🔄 Refresh Data", use_container_width=True,
                     disabled=not _fetcher_ok):
            with st.spinner("Fetching latest AAPL data..."):
                try:
                    fetch_and_update()
                    st.success("Data refreshed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Fetch failed:\n{e}")
        if not _fetcher_ok:
            st.caption("⚠️ live_fetcher not available")

    # 2. Retrain model
    with col2:
        st.markdown("**Step 2**")
        if st.button("🚀 Retrain Model", use_container_width=True,
                     disabled=not _retrain_ok):
            with st.spinner("Training on MPS/GPU... (~15-30s)"):
                try:
                    result = run_retrain(force=True)
                    st.success(
                        f"Done — val loss: {result['best_val_loss']:.6f}  "
                        f"({result['epochs_trained']} epochs)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Retrain failed:\n{e}")
        if not _retrain_ok:
            st.caption("⚠️ retrain_pipeline not available")

    # 3. Generate forecast
    with col3:
        st.markdown("**Step 3**")
        if st.button("🔮 Generate Forecast", use_container_width=True,
                     disabled=not _forecaster_ok):
            if not pt_files:
                st.error("No trained model found. Retrain first.")
            else:
                with st.spinner("Generating 7-day forecast..."):
                    try:
                        fc_df = generate_forecast()
                        st.success(
                            f"Forecast generated for "
                            f"{fc_df['date'].iloc[0].strftime('%b %d')} – "
                            f"{fc_df['date'].iloc[-1].strftime('%b %d')}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Forecast failed:\n{e}")
        if not _forecaster_ok:
            st.caption("⚠️ forecaster not available")

    # 4. Run evaluation
    with col4:
        st.markdown("**Step 4**")
        if st.button("📊 Run Evaluation", use_container_width=True,
                     disabled=not _eval_ok):
            with st.spinner("Comparing forecasts vs actuals..."):
                try:
                    new_records = run_evaluation()
                    if not new_records:
                        st.info(
                            "No new records. Either all forecasts are already "
                            "evaluated, or forecast dates haven't passed yet."
                        )
                    else:
                        st.success(f"Evaluated {len(new_records)} forecast run(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed:\n{e}")
        if not _eval_ok:
            st.caption("⚠️ comparator not available")

    st.divider()

    # ── Retrain history log ──
    retrain_records = _read_jsonl(RETRAIN_LOG)
    if retrain_records:
        st.subheader("Retrain History")
        rt_df = pd.DataFrame(retrain_records).tail(10)[
            ["run_date","model_version","best_val_loss",
             "epochs_trained","early_stopped","train_samples","val_samples"]
        ]
        st.dataframe(rt_df[::-1].reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation History
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    records = _read_jsonl(EVAL_LOG)

    if not records:
        st.info(
            "No evaluation records yet. Forecast dates must pass before actual "
            "prices are available for comparison. "
            "Once they do, click **Run Evaluation** in Model & Controls."
        )
        st.stop()

    eval_df = pd.DataFrame(records)

    # ── Summary metrics across all runs ──
    st.subheader("Overall Performance")

    completed = eval_df[eval_df.get("computation_complete", pd.Series([False]*len(eval_df)))]
    n_total   = len(eval_df)
    n_complete = int(eval_df.get("computation_complete", pd.Series([False]*len(eval_df))).sum()) if "computation_complete" in eval_df.columns else 0
    avg_dir   = float(eval_df["directional_accuracy"].mean()) if "directional_accuracy" in eval_df.columns else None
    avg_mae   = float(eval_df["mae_usd"].mean())              if "mae_usd" in eval_df.columns else None
    avg_rmse  = float(eval_df["rmse_usd"].mean())             if "rmse_usd" in eval_df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Runs Evaluated", n_total)
    c2.metric("Avg Directional Accuracy",
              f"{avg_dir:.1%}" if avg_dir is not None else "—",
              f"{'▲ above' if avg_dir and avg_dir > 0.5 else '▼ below'} 50% baseline"
              if avg_dir is not None else "")
    c3.metric("Avg MAE",  f"${avg_mae:.2f}"  if avg_mae  is not None else "—")
    c4.metric("Avg RMSE", f"${avg_rmse:.2f}" if avg_rmse is not None else "—")

    if avg_dir is not None:
        if avg_dir >= 0.55:
            st.success(
                f"Directional accuracy {avg_dir:.1%} — above 55% threshold. "
                "The model appears to be capturing a real signal."
            )
        elif avg_dir >= 0.50:
            st.warning(
                f"Directional accuracy {avg_dir:.1%} — marginally above coin-flip. "
                "More data needed to confirm a real signal."
            )
        else:
            st.error(
                f"Directional accuracy {avg_dir:.1%} — below 50% baseline. "
                "Consider retraining on more recent data."
            )

    st.divider()

    # ── Directional accuracy bar chart ──
    tail_10 = eval_df.tail(10).copy()

    if "directional_accuracy" in tail_10.columns:
        st.subheader("Directional Accuracy — Last 10 Runs")

        fig_dir = go.Figure()
        colors  = [
            "#2ecc71" if v >= 0.55 else ("#f39c12" if v >= 0.50 else "#e74c3c")
            for v in tail_10["directional_accuracy"]
        ]
        fig_dir.add_trace(go.Bar(
            x=tail_10["run_timestamp"].str[:10],
            y=tail_10["directional_accuracy"],
            marker_color=colors,
            text=[f"{v:.1%}" for v in tail_10["directional_accuracy"]],
            textposition="outside",
        ))
        fig_dir.add_hline(
            y=0.5, line_dash="dash", line_color="red",
            annotation_text="Coin-flip baseline (50%)",
            annotation_position="top right",
        )
        fig_dir.add_hline(
            y=0.55, line_dash="dot", line_color="rgba(46,204,113,0.5)",
            annotation_text="Signal threshold (55%)",
            annotation_position="bottom right",
        )
        fig_dir.update_yaxes(tickformat=".0%", range=[0, 1.1])
        fig_dir.update_layout(
            xaxis_title="Forecast Run Date",
            yaxis_title="Directional Accuracy",
            template="plotly_dark", height=360,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_dir, use_container_width=True)

    # ── MAE / RMSE over time ──
    if "mae_usd" in tail_10.columns and "rmse_usd" in tail_10.columns:
        st.subheader("Error Metrics — Last 10 Runs")

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=tail_10["run_timestamp"].str[:10],
            y=tail_10["mae_usd"],
            mode="lines+markers", name="MAE (USD)",
            line=dict(color="#3498db", width=2),
        ))
        fig_err.add_trace(go.Scatter(
            x=tail_10["run_timestamp"].str[:10],
            y=tail_10["rmse_usd"],
            mode="lines+markers", name="RMSE (USD)",
            line=dict(color="#e67e22", width=2, dash="dash"),
        ))
        fig_err.update_layout(
            xaxis_title="Forecast Run Date",
            yaxis_title="Error (USD)",
            template="plotly_dark", height=300,
            legend=dict(orientation="h"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Full evaluation table ──
    st.subheader("Evaluation Log — Last 10 Runs")
    display_cols = [
        "run_timestamp", "model_version", "horizon_available",
        "mae_usd", "rmse_usd", "directional_accuracy", "computation_complete",
    ]
    display_cols = [c for c in display_cols if c in tail_10.columns]
    fmt = tail_10[display_cols].copy()
    if "mae_usd"  in fmt.columns: fmt["mae_usd"]  = fmt["mae_usd"].map(lambda v: f"${v:.2f}")
    if "rmse_usd" in fmt.columns: fmt["rmse_usd"] = fmt["rmse_usd"].map(lambda v: f"${v:.2f}")
    if "directional_accuracy" in fmt.columns:
        fmt["directional_accuracy"] = fmt["directional_accuracy"].map(lambda v: f"{v:.1%}")
    fmt = fmt[::-1].reset_index(drop=True)
    fmt.index = range(1, len(fmt) + 1)
    st.dataframe(fmt, use_container_width=True)

    st.caption(
        "Directional accuracy: fraction of days where predicted direction "
        "(up/down vs last known price) matched actual direction. "
        "Above 55% over 30+ runs suggests real learned signal."
    )
