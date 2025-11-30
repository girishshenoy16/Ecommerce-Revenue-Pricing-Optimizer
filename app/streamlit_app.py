import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from scipy.stats import ks_2samp
import shap
import sys

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data_loader import load_raw_transactions, build_daily_aggregates, load_processed
from app.forecasting import train_revenue_model, forecast_with_scenario, FEATURE_COLS, load_revenue_model
from app.pricing import estimate_price_elasticity, recommend_price, load_elasticity
from app.insights import top_categories_by_revenue, promo_effect, daily_summary, data_quality_report
from app.drift_utils import calculate_psi



BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="E-Commerce Revenue & Pricing Analytics",
    layout="wide"
)

@st.cache_data
def get_data():
    raw_df = load_raw_transactions()
    daily_df = build_daily_aggregates(raw_df)
    return raw_df, daily_df

def main():
    st.title("📈 E-Commerce Revenue Forecasting & Pricing Optimization")

    st.sidebar.header("Controls")
    raw_df, daily_df = get_data()


    with st.sidebar.expander("Model Training & Elasticity"):
        if st.button("Train Revenue Model"):
            metrics = train_revenue_model()
            st.success(f"Model trained.")
            st.success(f"MAE: {metrics['MAE']:.2f}")
            st.success(f"RMSE: {metrics['RMSE']:.2f}")
            st.success(f"R² Score: {metrics['R2']:.3f}")
            st.success(f"MAPE %: {metrics['MAPE']:.2f}")

        if st.button("Estimate Price Elasticity"):
            elasticity = estimate_price_elasticity()
            st.json(elasticity)

        
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧹 Data Quality Check",
        "📈 Model Evaluation",
        "📊 Historical Analytics",
        "🔮 Forecast Simulator",
        "💰 Pricing Optimizer",
        "🧠 Explainability (SHAP)",
        "⚠️ Model Drift Detection"
    ])

    # ---------------- Tab 1: Data Quality Check ----------------
    with tab1:
        st.subheader("🧹 Data Quality & Cleaning Summary")

        st.write("This section shows how the raw dataset looks *after cleaning*.")

        dq = data_quality_report(raw_df)

        colA, colB = st.columns(2)

        with colA:
            st.metric("Total Rows (after cleaning)", dq["rows"])
            st.metric("Date Range", f"{dq['min_date']} → {dq['max_date']}")
            st.metric("Number of Categories", dq["num_categories"])

        with colB:
            st.write("Missing Values Per Column")
            st.json(dq["missing_values"])

            st.write("Duplicate Rows ", dq['duplicate_rows'])

        st.write("Preview of Cleaned Data")
        st.dataframe(raw_df.head())

    # ---------------- Tab 2: Model Evaluation ----------------
    with tab2:
        try:
            st.subheader("📈 Model Evaluation Reports")

            reports_path = Path("reports/visuals")

            # ----------------------------
            # 1. Actual vs Predicted
            # ----------------------------
            img1 = reports_path / "actual_vs_predicted.png"
            if img1.exists():
                st.markdown("### 📌 Actual vs Predicted Revenue")
                st.image(str(img1))
            else:
                st.warning("Run 'Train Revenue Model' first to generate this plot.")

            # ----------------------------
            # 2. Residual Distribution
            # ----------------------------
            img2 = reports_path / "residual_distribution.png"
            if img2.exists():
                st.markdown("### 📌 Residual Distribution")
                st.image(str(img2))
            else:
                st.warning("Residual plot not found. Train model again.")

            # ----------------------------
            # 3. Feature Importance
            # ----------------------------
            img3 = reports_path / "feature_importance.png"
            if img3.exists():
                st.markdown("### 📌 Feature Importance")
                st.image(str(img3))
            else:
                st.warning("Feature importance plot missing. Train model again.")

            # ----------------------------
            # 4. Error Over Time
            # ----------------------------
            img4 = reports_path / "error_over_time.png"
            if img4.exists():
                st.markdown("### 📌 Error Over Time")
                st.image(str(img4))
            else:
                st.warning("Error over time plot unavailable. Train model again.")
        
        except Exception as e:
            st.warning("Please train the revenue model first to enable this feature.")
            st.error(f"Error running forecast: {e}")


    # ---------------- Tab 3: Historical Analytics ----------------
    with tab3:
        st.subheader("Historical Performance")

        col1, col2 = st.columns(2)

        with col1:
            stats = daily_summary(daily_df)
            st.metric("Avg Daily Revenue", f"{stats['avg_daily_revenue']:.2f}")
            st.metric("Max Daily Revenue", f"{stats['max_daily_revenue']:.2f}")
            st.metric("Min Daily Revenue", f"{stats['min_daily_revenue']:.2f}")

        with col2:
            st.write("Top Revenue-Driving Categories")
            top_cats = top_categories_by_revenue(raw_df)
            st.dataframe(top_cats)

            st.write("Promo Effect on Units & Revenue")
            promo_df = promo_effect(raw_df)
            st.dataframe(promo_df)

        fig_rev = px.line(daily_df, x="date", y="total_revenue", title="Daily Revenue Trend")
        st.plotly_chart(fig_rev, use_container_width=True)


    # ---------------- Tab 4: Forecast Simulator ----------------
    with tab4:
        st.subheader("Revenue Forecast Simulator")

        future_days = st.slider("Forecast horizon (days)", 7, 60, 14)
        discount_shift = st.slider("Change in avg discount (%)", -20, 20, 0) / 100
        promo_shift = st.slider("Change in promo share (percentage points)", -30, 30, 0) / 100

        if st.button("Run Forecast Scenario"):
            try:
                future_df = forecast_with_scenario(
                    daily_df,
                    future_days=future_days,
                    discount_shift=discount_shift,
                    promo_shift=promo_shift,
                )

                st.write("Forecasted Revenue under Scenario")
                st.dataframe(future_df[["date", "predicted_revenue"]])

                combined = pd.concat(
                    [
                        daily_df[["date", "total_revenue"]].tail(60).rename(columns={"total_revenue": "revenue"}),
                        future_df[["date", "predicted_revenue"]].rename(columns={"predicted_revenue": "revenue"}),
                    ],
                    ignore_index=True
                )

                combined["type"] = ["History"] * 60 + ["Forecast"] * len(future_df)

                fig = px.line(
                    combined,
                    x="date",
                    y="revenue",
                    color="type",
                    title="Revenue History vs Forecast"
                )

                st.plotly_chart(fig, use_container_width=True)

                baseline_future = forecast_with_scenario(daily_df, future_days=future_days)
                uplift = future_df["predicted_revenue"].sum() - baseline_future["predicted_revenue"].sum()
                st.info(f"Total revenue change vs baseline over {future_days} days: {uplift:,.2f}")

            except Exception as e:
                st.warning("Please train the revenue model first to enable this feature.")
                st.error(f"Error running forecast: {e}")


    # ---------------- Tab 5: Pricing Optimizer ----------------
    with tab5:
        try:
            st.subheader("Dynamic Pricing Recommendations")

            cats = sorted(raw_df["category"].unique())
            category = st.selectbox("Select category", cats)

            elasticity = load_elasticity()
            
            if not elasticity:
                st.warning("Elasticity not estimated yet. Click 'Estimate Price Elasticity' in sidebar.")
            else:
                st.write("Estimated elasticity:", round(elasticity.get(category, "N/A"), 4))

            colp1, colp2 = st.columns(2)
            
            with colp1:
                current_price = st.number_input("Current price", min_value=1.0, max_value=10000.0, value=1000.0, step=10.0)
                cost_price = st.number_input("Cost price (optional, for margin)", min_value=0.0, max_value=9999.0, value=0.0, step=10.0)
            
            with colp2:
                target_change_pct = st.slider(
                    "Target change in demand (%)",
                    -50, 100, 10
                ) / 100.0

            if st.button("Recommend Price"):
                result = recommend_price(
                    current_price=current_price,
                    category=category,
                    target_change=target_change_pct,
                    cost_price=cost_price if cost_price > 0 else None,
                )

                st.success(f"Suggested price: {result['suggested_price']}")
                st.write("Elasticity used:", round((result["elasticity_used"]), 4))

                if result["margin_info"]:
                    mi = result["margin_info"]
                    st.write(f"Old margin: {mi['old_margin']:.2f}")
                    st.write(f"New margin:    {mi['new_margin']:.2f}")
                    st.write(f"Margin change:   {mi['margin_change']:.2f}")

                st.caption("Note: Simplified economic model for educational purposes.")

        except Exception as e:
            st.warning("Please train the revenue model first to enable this feature.")
            st.error(f"Error running forecast: {e}")


    # ---------------- Tab 6: Explainability (SHAP) ----------------
    with tab6:
        try:
            st.subheader("🧠 Model Explainability using SHAP")

            st.write("""
                SHAP (SHapley Additive exPlanations) helps us understand
                how each feature contributes to the model's revenue prediction.
            """)

            # Load model
            model = load_revenue_model()
            daily_df = load_processed()

            X = daily_df[[
                "day_of_week",
                "week_of_year",
                "month",
                "avg_discount",
                "promo_share",
                "total_units"
            ]]

            # ---- SHAP Explainer ----
            st.info("Computing SHAP values... (may take 5–10 seconds)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # ---- 1. SHAP Summary Plot (Beeswarm) ----
            st.markdown("### 📌 SHAP Summary Plot (Beeswarm)")

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig1)

            # ---- 2. SHAP Bar Plot (Mean Abs SHAP Value) ----
            st.markdown("### 📌 Feature Importance (SHAP - Mean Impact)")

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig2)

            # ---- 3. Force Plot (Single Prediction) ----
            st.markdown("### 📌 Force Plot (Single Day Prediction Explanation)")

            idx = st.slider("Pick a day index", 0, len(X)-1, 0)

            shap_value_single = shap_values[idx]

            st.write("Explanation for date:", str(daily_df['date'].iloc[idx]))

            plt.figure(figsize=(12, 2))
            shap.force_plot(
                explainer.expected_value,
                shap_value_single,
                X.iloc[idx],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()

            st.success("SHAP explainability loaded successfully!")

                # ================================================================
            # 4. SHAP Waterfall Plot (Single Prediction)
            # ================================================================
            st.markdown("### 🔥 SHAP Waterfall Plot (Single Prediction Breakdown)")

            idx_w = st.slider("Select index for waterfall", 0, len(X)-1, 5, key="waterfall_idx")

            feature_values = X.iloc[idx_w]
            shap_vals_single = shap_values[idx_w]

            st.write("Waterfall explanation for date:", str(daily_df['date'].iloc[idx_w]))

            fig_w = plt.figure(figsize=(10, 6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[0],
                shap_vals_single,
                feature_values,
                max_display=10
            )
            st.pyplot(fig_w)
            plt.close()

            st.info("The waterfall plot shows how each feature pushes the prediction up/down.")


            # ================================================================
            # 5. SHAP Comparison: Compare Two Different Days
            # ================================================================
            st.markdown("## 🆚 SHAP Comparison Between Two Dates")

            colA, colB = st.columns(2)

            with colA:
                idx_A = st.number_input("Pick first day index (A)", min_value=0, max_value=len(X)-1, value=0)
                st.write("Date A:", str(daily_df['date'].iloc[idx_A]))

            with colB:
                idx_B = st.number_input("Pick second day index (B)", min_value=0, max_value=len(X)-1, value=1)
                st.write("Date B:", str(daily_df['date'].iloc[idx_B]))

            shap_A = shap_values[idx_A]
            shap_B = shap_values[idx_B]

            pred_A = model.predict([X.iloc[idx_A]])[0]
            pred_B = model.predict([X.iloc[idx_B]])[0]

            # Output the difference
            st.write("### 📌 Prediction Comparison:")
            st.write(f"Prediction A: **{pred_A:,.2f}**")
            st.write(f"Prediction B: **{pred_B:,.2f}**")
            st.write(f"🔺 Difference: **{pred_B - pred_A:,.2f}**")

            # Show the contribution difference table
            st.markdown("### 🧩 Feature Contribution Differences")

            contrib_df = pd.DataFrame({
                "Feature": X.columns,
                "SHAP_A": shap_A,
                "SHAP_B": shap_B,
                "Contribution_Diff": shap_B - shap_A
            })

            st.dataframe(contrib_df.sort_values("Contribution_Diff", ascending=False))


            # ================================================================
            # 6. Side-by-side Waterfall Comparison
            # ================================================================
            st.markdown("### 🔥 Side-by-Side SHAP Waterfall Comparison")

            colW1, colW2 = st.columns(2)

            with colW1:
                st.markdown(f"#### Waterfall A - {daily_df['date'].iloc[idx_A]}")
                fig_WA = plt.figure(figsize=(9, 5))
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[0],
                    shap_A,
                    X.iloc[idx_A],
                    max_display=10
                )
                st.pyplot(fig_WA)
                plt.close()

            with colW2:
                st.markdown(f"#### Waterfall B - {daily_df['date'].iloc[idx_B]}")
                fig_WB = plt.figure(figsize=(9, 5))
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[0],
                    shap_B,
                    X.iloc[idx_B],
                    max_display=10
                )
                st.pyplot(fig_WB)
                plt.close()

            st.success("SHAP explainability with comparison is fully loaded!")
        
        except Exception as e:
            st.warning("Please train the revenue model first to enable this feature.")
            st.error(f"Error running forecast: {e}")


    # ---------------- Tab 7: Model Drift Detection ----------------
    with tab7:
        try:
            st.subheader("⚠️ Model Drift Detection")

            st.write("""
                Model drift occurs when the data used for prediction changes compared to 
                the data used for training. This dashboard detects drift in:
                - Feature distributions
                - Target (revenue) distribution
                - Model prediction drift
                - Statistical drift using KS-test
            """)


            # Load cleaned full dataset
            df_full = load_processed().sort_values("date")

            # Determine training vs recent window
            train_df = df_full.iloc[:-30]      # everything except last 30 days
            recent_df = df_full.iloc[-30:]     # last 30 days

            model = load_revenue_model()

            # Predictions
            train_pred = model.predict(train_df[FEATURE_COLS])
            recent_pred = model.predict(recent_df[FEATURE_COLS])

            st.markdown("## 📌 Drift Summary Table")

            drift_rows = []

            for feature in FEATURE_COLS + ["total_revenue"]:
                # KS Test
                stat, pvalue = ks_2samp(train_df[feature], recent_df[feature])

                psi = calculate_psi(train_df[feature], recent_df[feature])

                drift_rows.append({
                    "Feature": feature,
                    "PSI": round(psi, 4),
                    "Drift": (
                        "Stable ✅"      if psi < 0.1 else
                        "Mild ⚠️"       if psi < 0.25 else
                        "Severe 🚨"
                    )
                })

            # Prediction drift
            stat_pred, p_pred = ks_2samp(train_pred, recent_pred)
            drift_rows.append({
                "Feature": "MODEL_PREDICTION",
                "Drift_PValue": round(p_pred, 4),
                "Drift_Detected": "YES ⚠️" if p_pred < 0.05 else "NO ✅"
            })

            st.dataframe(drift_rows)

            # =============================================
            # VISUALIZATIONS
            # =============================================

            st.markdown("## 📉 Feature Distribution Drift")

            for feature in FEATURE_COLS:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(train_df[feature], bins=30, alpha=0.5, label="Training Data")
                ax.hist(recent_df[feature], bins=30, alpha=0.5, label="Recent Data")
                ax.set_title(f"{feature} Drift Comparison")
                ax.legend()
                st.pyplot(fig)
                plt.close()

            # =============================================
            # Target Drift
            # =============================================
            st.markdown("## 💵 Revenue (Target) Distribution Drift")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(train_df["total_revenue"], bins=30, alpha=0.5, label="Training")
            ax.hist(recent_df["total_revenue"], bins=30, alpha=0.5, label="Recent")
            ax.set_title("Revenue Distribution Drift")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # =============================================
            # Prediction Drift
            # =============================================
            st.markdown("## 🤖 Prediction Drift (Model Output)")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(train_pred, bins=30, alpha=0.5, label="Training Predictions")
            ax.hist(recent_pred, bins=30, alpha=0.5, label="Recent Predictions")
            ax.set_title("Prediction Drift Detection")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # =============================================
            # Drift Severity Warning
            # =============================================
            if p_pred < 0.01:
                st.error("🚨 Significant model drift detected! Model may need retraining.")
            else:
                st.success("✅ No significant drift detected in model predictions.")
        
        except Exception as e:
            st.warning("Please train the revenue model first to enable this feature.")
            st.error(f"Error running forecast: {e}")



if __name__ == "__main__":
    main()