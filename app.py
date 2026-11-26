# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import skew

st.set_page_config(page_title="Fraud Detection — Isolation Forest", layout="wide")
sns.set(style="whitegrid")

# --------------------------
# Utility functions (from your notebook)
# --------------------------
def convert_cat(features, df):
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")
    return df

def handle_skewed_features(
    df,
    zero_threshold=0.9,
    skew_threshold=0.5,
    num_features=None,
    exclude_cols=None
):
    """
    Handle skewed numerical features by applying appropriate transformations,
    *forcing* certain columns to be transformed even if they don't exceed skew_threshold.
    Returns: df, transformed_cols, high_zero_cols, auto_skewed, pt_dict
    """
    df = df.copy()
    if num_features is None:
        raise ValueError("`num_features` must be provided")
    if exclude_cols is None:
        exclude_cols = []

    numerical_cols = [c for c in num_features if c not in exclude_cols]

    # detect ultra-sparse
    zero_ratios = (df[numerical_cols] == 0).sum() / len(df)
    high_zero_cols = zero_ratios[zero_ratios > zero_threshold].index.tolist()

    # compute skew
    skew_vals = df[numerical_cols].apply(lambda s: skew(s.dropna()) if s.dropna().shape[0] > 0 else 0)
    auto_skewed = skew_vals[abs(skew_vals) > skew_threshold].index.tolist()

    # union these with your forced list
    to_transform = list(set(auto_skewed))
    transformed_cols = []
    dropped_cols     = []
    pt_dict          = {}

    for col in to_transform:
        if col in high_zero_cols:
            # binary indicator + log
            df[f"Has_{col}"] = (df[col] > 0).astype(int)
            df[f"Log_{col}"] = df[col].map(lambda x: np.log1p(x) if x > 0 else 0)
            transformed_cols += [f"Has_{col}", f"Log_{col}"]
            dropped_cols.append(col)
        elif df[col].nunique() <= 5:
            # skip transformation for small-cardinality columns
            continue
        else:
            pt = PowerTransformer(method="yeo-johnson")
            arr = df[[col]].values
            df[f"PT_{col}"] = pt.fit_transform(arr)
            pt_dict[col] = pt
            transformed_cols.append(f"PT_{col}")
            dropped_cols.append(col)

    # drop originals for transformed columns
    df.drop(columns=dropped_cols, inplace=True, errors='ignore')

    return df, transformed_cols, high_zero_cols, auto_skewed, pt_dict

# --------------------------
# Session state containers
# --------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None        # the original uploaded dataframe (unchanged)
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None     # original + prediction cols (used for visuals)
if "X_use" not in st.session_state:
    st.session_state.X_use = None         # model features used for training/pca
if "iso_model" not in st.session_state:
    st.session_state.iso_model = None     # trained IsolationForest
if "proc_df_internal" not in st.session_state:
    st.session_state.proc_df_internal = None  # processed dataframe used for predictions

# --------------------------
# Sidebar navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload Data", "Detect Anomalies", "Visualizations"])

# --------------------------
# PAGE: Upload Data
# --------------------------
if page == "Upload Data":
    st.title("📁 Upload Bank Transactions CSV")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.raw_df = df.copy()
        # reset any previous results
        st.session_state.merged_df = None
        st.session_state.X_use = None
        st.session_state.iso_model = None
        st.session_state.proc_df_internal = None

        st.success("Dataset uploaded.")
        st.write("Dataset preview:")
        st.dataframe(df.head())
        st.write("Columns:", list(df.columns))

# --------------------------
# PAGE: Detect Anomalies
# --------------------------
elif page == "Detect Anomalies":
    st.title("🛡️ Detect Anomalies (Isolation Forest)")

    if st.session_state.raw_df is None:
        st.warning("Upload dataset first on the 'Upload Data' page.")
    else:
        # Work on a copy of raw data
        orig_df = st.session_state.raw_df.copy()
        df = orig_df.copy()

        st.subheader("Running preprocessing from notebook...")

        # --- Date conversions & chronological sorting ---
        if "TransactionDate" not in df.columns:
            st.error("TransactionDate column not found in uploaded file. Preprocessing cannot continue.")
        else:
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
            # sort by AccountID then date if AccountID exists, else just by date
            if "AccountID" in df.columns:
                df = df.sort_values(["AccountID", "TransactionDate"])
                df["TimeSinceLastTx"] = df.groupby("AccountID")["TransactionDate"].diff().dt.total_seconds().fillna(0)
            else:
                df = df.sort_values("TransactionDate")
                df["TimeSinceLastTx"] = df["TransactionDate"].diff().dt.total_seconds().fillna(0)

            df["Hour"] = df["TransactionDate"].dt.hour
            df["Month"] = df["TransactionDate"].dt.month
            # Notebook used .dt.day here (not dayofweek)
            df["DayOfWeek"] = df["TransactionDate"].dt.day
            # Unix time - safe cast
            try:
                df["UnixTime"] = df["TransactionDate"].astype("int64") // 10**9
            except Exception:
                df["UnixTime"] = df["TransactionDate"].view(np.int64) // 10**9

            # Cyclic encodings
            df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
            df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
            df["Day_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
            df["Day_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

            # Drop helper columns if present
            for c in ["PreviousTransactionDate", "TransactionDate"]:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)

        # --- Convert categorical features ---
        new_cat_features = ["TransactionType", "Channel", "Region", "CustomerOccupation", "LoginAttempts"]
        df = convert_cat(new_cat_features, df)

        # --- Numeric features as in your notebook ---
        num_features_raw = [
            "TransactionAmount",
            "CustomerAge",
            "TransactionDuration",
            "AccountBalance",
            "TimeSinceLastTx",
            "Hour",
            "DayOfWeek",
            "UnixTime",
            "Month",
        ]

        # Safety: ensure missing numeric cols are created (so code doesn't break) and notify user
        for col in num_features_raw:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found. Creating '{col}' filled with zeros to continue.")
                df[col] = 0

        # --- Handle skewed features (PowerTransformer, log, Has_ for sparse) ---
        with st.spinner("Applying skew-handling transforms..."):
            df_processed_internal, transformed_cols, sparse_cols, skew_cols, pt_dict = handle_skewed_features(
                df=df,
                num_features=num_features_raw
            )

        # Build final feature matrix X_use (the features used for model training and PCA)
        desired_features = [
            "PT_TransactionAmount",
            "CustomerAge",
            "PT_TransactionDuration",
            "PT_AccountBalance",
            "Hour_sin",
            "Hour_cos",
            "Day_sin",
            "Day_cos",
            "PT_TimeSinceLastTx",
            "Hour",
            "DayOfWeek",
            "UnixTime",
            "Month",
        ]

        X_use = pd.DataFrame(index=df_processed_internal.index)
        for feat in desired_features:
            if feat in df_processed_internal.columns:
                X_use[feat] = df_processed_internal[feat]
            else:
                # fallback: if PT_ missing, use raw column; otherwise create zero column
                if feat.startswith("PT_"):
                    raw = feat.replace("PT_", "")
                    if raw in df_processed_internal.columns:
                        X_use[feat] = df_processed_internal[raw]
                    else:
                        X_use[feat] = 0.0
                else:
                    X_use[feat] = df_processed_internal.get(feat, 0.0)

        # Fill NaN/inf to be safe
        X_use = X_use.replace([np.inf, -np.inf], np.nan).fillna(0)

        st.success("Preprocessing complete. Features ready.")
        st.write("Feature preview (first rows):")
        st.dataframe(X_use.head())

        # --- Train Isolation Forest ---
        st.subheader("Train Isolation Forest")
        contamination = st.slider("Expected anomaly rate (contamination)", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
        n_estimators = st.number_input("n_estimators", value=100, step=10)

        if st.button("Train Isolation Forest"):
            iso_forest = IsolationForest(n_estimators=int(n_estimators), contamination=float(contamination), random_state=42)
            iso_forest.fit(X_use)
            st.session_state.iso_model = iso_forest
            # store internal processed df used for transformations (may lack original columns)
            st.session_state.proc_df_internal = df_processed_internal.copy()
            st.session_state.X_use = X_use.copy()

            # --- Predict on X_use and create result columns in the internal processed df ---
            df_internal = df_processed_internal.copy().reset_index(drop=True)
            scores = iso_forest.decision_function(X_use)
            preds = iso_forest.predict(X_use)

            df_internal["AnomalyScore"] = scores
            df_internal["IsAnomaly"] = preds
            df_internal["AnomalyLabel"] = df_internal["IsAnomaly"].map({1: "Normal", -1: "Potential Fraud"})

            # --- MERGE ORIGINAL RAW DATA BACK FOR VISUALIZATION ---
            orig = orig_df.reset_index(drop=True)
            proc = df_internal.reset_index(drop=True)

            if len(orig) != len(proc):
                st.error("❌ ERROR: Row mismatch between original and processed data. Visualizations cannot be produced reliably.")
                # still store internal results so user can download
                st.session_state.merged_df = None
            else:
                merged = orig.copy()
                # add prediction columns (overwrite if exist)
                merged["AnomalyScore"] = proc["AnomalyScore"]
                merged["IsAnomaly"] = proc["IsAnomaly"]
                merged["AnomalyLabel"] = proc["AnomalyLabel"]

                # store merged dataframe for visualization & downloads
                st.session_state.merged_df = merged
                st.session_state.proc_df_internal = df_internal  # keep internal as well

            st.success("Isolation Forest trained and results merged with original dataset (session stored).")

            # Show detection summary & download buttons
            if st.session_state.merged_df is not None:
                n_anom = (st.session_state.merged_df["IsAnomaly"] == -1).sum()
                st.write(f"Total records: {len(st.session_state.merged_df)}, anomalies detected: {n_anom}")
                st.dataframe(st.session_state.merged_df.head())

                st.download_button("Download full results (CSV)", st.session_state.merged_df.to_csv(index=False), "full_predictions.csv", "text/csv")
                anomalies = st.session_state.merged_df[st.session_state.merged_df["IsAnomaly"] == -1]
                st.download_button("Download anomalies (CSV)", anomalies.to_csv(index=False), "anomalies.csv", "text/csv")
            else:
                # fallback: provide internal processed results for download
                st.write("Merged dataset unavailable due to mismatch, but you can download internal processed results.")
                st.download_button("Download internal processed results CSV", df_internal.to_csv(index=False), "internal_results.csv", "text/csv")

# --------------------------
# PAGE: Visualizations
# --------------------------
elif page == "Visualizations":
    st.title("📊 Visualizations")

    # require trained model and merged data
    if st.session_state.iso_model is None or st.session_state.merged_df is None or st.session_state.X_use is None:
        st.warning("Please run 'Detect Anomalies' and train the model first (and ensure there was no row mismatch).")
    else:
        merged_df = st.session_state.merged_df.copy()
        Xv = st.session_state.X_use.copy()
        iso_forest = st.session_state.iso_model

        # Ensure expected columns exist for plotting; insert placeholders if missing
        for c in ["TransactionAmount", "TransactionDuration", "CustomerAge", "AccountBalance", "TransactionType", "Channel", "CustomerOccupation", "Region", "Month"]:
            if c not in merged_df.columns:
                merged_df[c] = np.nan if c in ["TransactionAmount", "TransactionDuration", "CustomerAge", "AccountBalance"] else "Unknown"

        # PCA 2D
        st.subheader("PCA 2D — Isolation Forest clusters")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(Xv)
        merged_df = merged_df.reset_index(drop=True)
        merged_df["PCA1_Isolation"] = X_pca[:, 0]
        merged_df["PCA2_Isolation"] = X_pca[:, 1]

        normal_df = merged_df[merged_df["IsAnomaly"] == 1]
        anomaly_df = merged_df[merged_df["IsAnomaly"] == -1]

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        ax[0].scatter(normal_df["PCA1_Isolation"], normal_df["PCA2_Isolation"],
                      s=30, alpha=0.3, label=f'Normal (n={len(normal_df)})', c="#A0A0A0")
        ax[0].scatter(anomaly_df["PCA1_Isolation"], anomaly_df["PCA2_Isolation"],
                      s=50, alpha=0.7, label=f'Potential Fraud (n={len(anomaly_df)})',
                      c="red", edgecolors="black", linewidths=0.5)
        ax[0].set_title("Isolation Forest — Potential Frauds vs Normal (PCA 2D)", fontsize=13, weight="bold", pad=15)
        ax[0].set_xlabel("PCA Component 1")
        ax[0].set_ylabel("PCA Component 2")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        # Histogram of anomaly scores
        threshold = 0
        sns.histplot(merged_df["AnomalyScore"], bins=60, kde=True, edgecolor="white", alpha=0.85, ax=ax[1])
        ax[1].axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")
        ax[1].text(threshold + 0.02, ax[1].get_ylim()[1]*0.8, "Normal zone →", color="green", fontsize=11)
        ax[1].text(threshold - 0.02, ax[1].get_ylim()[1]*0.8, "← Anomaly zone", color="red", fontsize=11, ha="right")
        ax[1].set_title("Distribution of Isolation Forest Anomaly Scores", fontsize=13, weight="bold", pad=15)
        ax[1].set_xlabel("Anomaly Score (higher = more normal, lower = more anomalous)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Fraud Behavior Dashboard (3×3)")

        def plot_cluster_features_isolation_forest(df, month_col="Month"):
            # ensure categorical placeholders exist
            for c in ["TransactionType","Channel","CustomerOccupation","Region"]:
                if c not in df.columns:
                    df[c] = "Unknown"

            normal_data = df[df["IsAnomaly"] == 1].copy()
            fraud_data = df[df["IsAnomaly"] == -1].copy()

            fig, ax = plt.subplots(3, 3, figsize=(24, 20))
            ax = ax.flatten()

            # 1) TransactionAmount distribution
            sns.boxplot(data=df, x="AnomalyLabel", y="TransactionAmount", palette="viridis", ax=ax[0])
            ax[0].set_title("Distribution of TransactionAmount by Anomaly Label")

            # 2) TransactionType counts
            order = df["TransactionType"].astype(str).unique().tolist()
            sns.countplot(data=df, x="TransactionType", palette="viridis", hue="AnomalyLabel", order=order, ax=ax[1])
            ax[1].set_title("Transaction Type Distribution")

            # 3) Channel counts
            order = df["Channel"].astype(str).unique().tolist()
            sns.countplot(data=df, x="Channel", palette="viridis", hue="AnomalyLabel", order=order, ax=ax[2])
            ax[2].set_title("Channel Distribution")

            # 4) CustomerAge distribution
            sns.boxplot(data=df, x="AnomalyLabel", y="CustomerAge", palette="viridis", ax=ax[3])
            ax[3].set_title("Distribution of CustomerAge by Anomaly Label")

            # 5) CustomerOccupation counts
            order = df["CustomerOccupation"].astype(str).unique().tolist()
            sns.countplot(data=df, x="CustomerOccupation", palette="viridis", hue="AnomalyLabel", order=order, ax=ax[4])
            ax[4].set_title("Customer Occupation Distribution")

            # 6) TransactionDuration distribution
            sns.boxplot(data=df, x="AnomalyLabel", y="TransactionDuration", palette="viridis", ax=ax[5])
            ax[5].set_title("Distribution of Transaction Duration by Anomaly Label")

            # 7) AccountBalance distribution
            sns.boxplot(data=df, x="AnomalyLabel", y="AccountBalance", palette="viridis", ax=ax[6])
            ax[6].set_title("Distribution of Account Balance by Anomaly Label")

            # 8) Region counts
            order = df["Region"].astype(str).unique().tolist()
            sns.countplot(data=df, x="Region", palette="viridis", hue="AnomalyLabel", order=order, ax=ax[7])
            ax[7].set_title("Region Distribution")

            # 9) Monthly trend: mean TransactionAmount
            # guard if month_col missing or not numeric
            if month_col not in df.columns:
                df[month_col] = np.nan
            try:
                trend = normal_data.groupby(month_col)["TransactionAmount"].mean().reset_index()
                fraud_trend = fraud_data.groupby(month_col)["TransactionAmount"].mean().reset_index()
                sns.lineplot(data=trend, x=month_col, y="TransactionAmount", ax=ax[8], label="Avg Normal Transactions", marker="o")
                sns.lineplot(data=fraud_trend, x=month_col, y="TransactionAmount", ax=ax[8], label="Avg Fraud Transactions", marker="X", color="red")
                if not fraud_trend.empty:
                    ax[8].fill_between(fraud_trend[month_col].values, fraud_trend["TransactionAmount"].values, color="crimson", alpha=0.15)
                ax[8].set_title("Monthly Trend: Avg Transaction Amount")
            except Exception:
                ax[8].text(0.5, 0.5, "Monthly trend unavailable", ha="center", va="center")
                ax[8].set_title("Monthly Trend: Avg Transaction Amount")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            return fig

        fig2 = plot_cluster_features_isolation_forest(merged_df)
        st.pyplot(fig2)
