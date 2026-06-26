from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# 1. Page Configuration MUST be the first Streamlit command called
st.set_page_config(
    page_title="Fraud Transaction Dashboard",
    page_icon="💰",
    layout="wide",
)

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_RESULTS_FILE = PROJECT_ROOT / "reports" / "model_results.csv"
CONFIG_FILE = Path(__file__).parent / "config.yaml"

# --- AUTHENTICATION SETUP ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# --- FIX: ONLY show login if NOT authenticated ---
# Check if user is already authenticated
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# Only show login form if not authenticated
if st.session_state["authentication_status"] is None or st.session_state["authentication_status"] is False:
    # Login form - only shown when not logged in
    try:
        authenticator.login(
            location="main",
            fields={
                "Form name": "Login",
                "Username": "Email",
                "Password": "Password",
                "Login": "Login",
            },
        )
    except TypeError:
        authenticator.login(
            "Login",
            "main",
            fields={
                "Form name": "Login",
                "Username": "Email",
                "Password": "Password",
                "Login": "Login",
            },
        )
    
    # Handle authentication status
    auth_status = st.session_state.get("authentication_status")
    
    if auth_status is False:
        st.error("Email or password is incorrect.")
        st.stop()
    elif auth_status is None:
        st.warning("Please log in to continue.")
        st.stop()
    else:
        # Successfully authenticated - rerun to clear login form
        st.rerun()

# --- CONTINUED ONLY IF LOGGED IN ---
# If we get here, user is authenticated, so login form won't be shown

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .section-note {
        color: #475569;
        font-size: 0.95rem;
        margin-top: -0.35rem;
        margin-bottom: 1.2rem;
    }
    .insight-box {
        border: 1px solid #dbe3ef;
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .small-label {
        color: #64748b;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stSidebar"] {
        background: #f5f7fb;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    .sidebar-title {
        color: #17468f;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sidebar-role {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 0;
        border-radius: 7px;
        color: #1f3f74;
        font-size: 0.92rem;
        font-weight: 600;
        justify-content: flex-start;
        margin: 0.12rem 0;
        padding: 0.72rem 0.85rem;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e8f1ff;
        color: #17468f;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #d8eaff;
        color: #123f85;
        font-weight: 800;
    }
    .sidebar-footer {
        border-top: 1px solid #d9e2ef;
        margin-top: 2rem;
        padding-top: 1.1rem;
    }
    .sidebar-pill {
        background: #17468f;
        border-radius: 7px;
        color: white;
        font-size: 0.9rem;
        font-weight: 700;
        padding: 0.8rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGES = {
    "overview": {
        "label": "Overview / Summary",
        "icon": "📊",
    },
    "eda": {
        "label": "EDA Insights",
        "icon": "🔍",
    },
    "dataset": {
        "label": "Dataset",
        "icon": "📋",
    },
    "models": {
        "label": "Model Comparison",
        "icon": "🤖",
    },
    "final-model": {
        "label": "Final Model",
        "icon": "⭐",
    },
}


@st.cache_data(show_spinner="Loading cleaned transaction data...")
def load_transactions() -> pd.DataFrame:
    use_columns = [
        "date",
        "amount",
        "use_chip",
        "merchant_state",
        "mcc",
        "errors",
        "is_fraud",
    ]
    data = pd.read_csv(DATA_FILE, usecols=use_columns)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["amount_value"] = pd.to_numeric(
        data["amount"].astype(str).str.replace("$", "", regex=False),
        errors="coerce",
    )
    data["abs_amount"] = data["amount_value"].abs()
    data["hour"] = data["date"].dt.hour
    data["has_error"] = data["errors"].notna().astype(int)
    data["use_chip"] = data["use_chip"].fillna("Unknown")
    data["merchant_state"] = data["merchant_state"].fillna("Unknown")
    data["mcc"] = data["mcc"].astype(str)
    data["is_fraud"] = data["is_fraud"].astype(int)
    return data


@st.cache_data(show_spinner="Loading model results...")
def load_model_results() -> pd.DataFrame:
    required_columns = {
        "model",
        "approach",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "training_time_seconds",
    }
    results = pd.read_csv(MODEL_RESULTS_FILE)
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in model_results.csv: {missing_text}")
    return results


def show_image(filename: str, caption: str) -> None:
    image_path = FIGURE_DIR / filename
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.info(f"Run the notebook that creates `{filename}` to display this chart.")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='section-note'>{subtitle}</p>", unsafe_allow_html=True)


def plot_model_metric(metric_columns: list[str], title: str, y_label: str):
    plot_data = MODEL_RESULTS.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=plot_data, x="metric", y="score", hue="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig


def show_summary_metrics(transactions: pd.DataFrame) -> None:
    total_transactions = len(transactions)
    fraud_transactions = int(transactions["is_fraud"].sum())
    fraud_rate = fraud_transactions / total_transactions * 100
    average_amount = transactions["abs_amount"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.4f}%")
    col4.metric("Average Amount", f"${average_amount:,.2f}")


def show_overview(transactions: pd.DataFrame) -> None:
    page_header(
        "Fraud Transaction Dashboard",
        "A compact prototype for transaction fraud analysis, model comparison and final model selection.",
    )

    show_summary_metrics(transactions)

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Project Focus")
        st.markdown(
            """
            <div class="insight-box">
            This dashboard supports fraud detection by showing transaction patterns,
            model performance and the final model choice. Because fraud is rare, recall,
            PR-AUC and fraud F1-score are treated as more meaningful than accuracy alone.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            transactions[
                ["date", "amount", "use_chip", "merchant_state", "mcc", "errors", "is_fraud"]
            ].head(50),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        show_image("eda_fraud_distribution.png", "Fraud vs non-fraud distribution")


def show_eda() -> None:
    page_header(
        "Exploratory Data Analysis",
        "Key fraud patterns from transaction amount, time, payment method and merchant categories.",
    )

    tab1, tab2, tab3 = st.tabs(["Core Patterns", "Merchant Risk", "Relationships"])

    with tab1:
        left, right = st.columns(2)
        with left:
            show_image("eda_fraud_rate_by_hour.png", "Fraud rate by transaction hour")
        with right:
            show_image("eda_fraud_rate_by_amount_range.png", "Fraud rate by amount range")

        left, right = st.columns(2)
        with left:
            show_image("eda_fraud_rate_by_transaction_method.png", "Fraud rate by transaction method")
        with right:
            show_image("eda_monthly_fraud_trend.png", "Monthly fraud rate trend")

    with tab2:
        left, right = st.columns(2)
        with left:
            show_image("eda_top_mcc_fraud_rate.png", "Top merchant categories by fraud rate")
        with right:
            show_image("eda_top_state_fraud_rate.png", "Top merchant states by fraud rate")

    with tab3:
        left, right = st.columns(2)
        with left:
            show_image("eda_amount_vs_hour_scatter.png", "Amount vs hour by fraud label")
        with right:
            show_image("eda_selected_feature_correlation.png", "Selected feature correlation")


def show_dataset(transactions: pd.DataFrame) -> None:
    page_header(
        "Dataset Explorer",
        "Select a transaction column to inspect its distribution, counts and missing values.",
    )

    column_options = {
        "Transaction Amount": "abs_amount",
        "Transaction Method": "use_chip",
        "Merchant State": "merchant_state",
        "Merchant Category Code": "mcc",
        "Transaction Hour": "hour",
        "Has Error": "has_error",
        "Fraud Label": "is_fraud",
    }
    selected_label = st.selectbox("Select a column to explore", list(column_options.keys()))
    selected_column = column_options[selected_label]
    column_data = transactions[selected_column].dropna()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Analysed", f"{len(transactions):,}")
    col2.metric("Missing Values", f"{transactions[selected_column].isna().sum():,}")
    col3.metric("Unique Values", f"{transactions[selected_column].nunique():,}")

    st.divider()
    left, right = st.columns([1.2, 1])

    if selected_column == "abs_amount":
        plot_data = column_data.sample(min(len(column_data), 100_000), random_state=42)
        with left:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            sns.histplot(plot_data, bins=50, ax=ax)
            ax.set_title("Distribution of Transaction Amount")
            ax.set_xlabel("Absolute amount")
            ax.set_ylabel("Transaction count")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with right:
            st.subheader("Amount Summary")
            st.dataframe(
                column_data.describe().rename("value").reset_index(),
                hide_index=True,
                use_container_width=True,
            )
    else:
        summary = (
            column_data.astype(str)
            .value_counts()
            .rename_axis("category")
            .reset_index(name="count")
        )
        summary["percentage"] = summary["count"] / summary["count"].sum() * 100
        chart_data = summary.head(10)

        with left:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            sns.barplot(data=chart_data, x="category", y="count", hue="category", legend=False, ax=ax)
            ax.set_title(f"Top Categories for {selected_label}")
            ax.set_xlabel(selected_label)
            ax.set_ylabel("Transaction count")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with right:
            st.subheader("Category Summary")
            st.dataframe(
                summary.head(20).assign(percentage=lambda data: data["percentage"].round(2)),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Sample Records")
    st.dataframe(
        transactions[["date", "amount", "use_chip", "merchant_state", "mcc", "errors", "is_fraud"]].head(30),
        hide_index=True,
        use_container_width=True,
    )


def show_models() -> None:
    page_header(
        "Model Comparison",
        "Performance comparison across supervised models and an unsupervised clustering model.",
    )

    st.dataframe(MODEL_RESULTS.round(4), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(
            plot_model_metric(
                ["accuracy", "roc_auc", "pr_auc"],
                "Overall Model Metrics",
                "Score",
            ),
            use_container_width=True,
        )
    with right:
        st.pyplot(
            plot_model_metric(
                ["fraud_precision", "fraud_recall", "fraud_f1"],
                "Fraud-Class Metrics",
                "Score",
            ),
            use_container_width=True,
        )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.barplot(
        data=MODEL_RESULTS,
        x="model",
        y="training_time_seconds",
        hue="model",
        legend=False,
        ax=ax,
    )
    ax.set_title("Training Time Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Training time (seconds)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Model interpretation notes"):
        st.markdown(
            """
            - **KNN** has high accuracy, but very low fraud recall, so it misses most fraud cases.
            - **K-Means** is unsupervised, so it is useful as a clustering comparison rather than a direct classifier.
            - **Random Forest** is selected for its fraud recall and feature-importance explanation.
            """
        )


def show_final_model() -> None:
    page_header(
        "Final Model",
        "Random Forest is selected because it balances fraud recall with explainability.",
    )

    rf_row = MODEL_RESULTS[MODEL_RESULTS["model"] == "Random Forest"]
    knn_row = MODEL_RESULTS[MODEL_RESULTS["model"] == "KNN"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Model", "Random Forest")
    col2.metric("Fraud Recall", f"{rf_row['fraud_recall'].iloc[0]:.4f}")
    col3.metric("PR-AUC", f"{rf_row['pr_auc'].iloc[0]:.4f}")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            """
            <div class="insight-box">
            <span class="small-label">Reason for selection</span><br>
            Random Forest is selected as the final model because it catches a high proportion
            of fraud transactions and provides feature importance, which makes the fraud-risk
            factors easier to explain in the report and viva.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.concat([rf_row, knn_row])[
                ["model", "accuracy", "pr_auc", "fraud_precision", "fraud_recall", "fraud_f1"]
            ].round(4),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            """
            **Why not KNN even though accuracy is high?**

            KNN accuracy is high because most transactions are non-fraud. Its fraud recall is
            very low, meaning it misses many true fraud cases. For fraud detection, recall and
            PR-AUC are more useful than accuracy alone.
            """
        )
    with right:
        show_image("model_random_forest_feature_importance.png", "Random Forest feature importance")

    st.subheader("Random Forest Evaluation Charts")
    left, right = st.columns(2)
    with left:
        show_image("model_random_forest_confusion_matrix.png", "Random Forest confusion matrix")
    with right:
        show_image("model_random_forest_precision_recall_curve.png", "Random Forest precision-recall curve")


def set_page(page_key: str) -> None:
    st.session_state["page"] = page_key


if not DATA_FILE.exists():
    st.error("Missing `data/processed/cleaned_transactions.csv`. Run `src/01_process_data.py` first.")
    st.stop()

if not MODEL_RESULTS_FILE.exists():
    st.error("Missing `reports/model_results.csv`. Run the model comparison notebook or restore the results CSV.")
    st.stop()

transactions = load_transactions()
MODEL_RESULTS = load_model_results()

with st.sidebar:
    # Only show logout button if authenticated
    if st.session_state["authentication_status"]:
        authenticator.logout("Logout", "sidebar")

    page = st.session_state.get("page", "overview")
    if page not in PAGES:
        page = "overview"

    st.markdown(
        """
        <div class="sidebar-title">Fraud Transaction Dashboard</div>
        <div class="sidebar-role">5011CEM Big Data Project</div>
        """,
        unsafe_allow_html=True,
    )

    for page_key, page_info in PAGES.items():
        button_label = f"{page_info['icon']}  {page_info['label']}"
        st.button(
            button_label,
            key=f"nav_{page_key}",
            type="primary" if page_key == page else "secondary",
            use_container_width=True,
            on_click=set_page,
            args=(page_key,),
        )

    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="sidebar-pill">Random Forest Selected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

if page == "overview":
    show_overview(transactions)
elif page == "eda":
    show_eda()
elif page == "dataset":
    show_dataset(transactions)
elif page == "models":
    show_models()
else:
    show_final_model()

st.divider()
st.caption(
    "Fraud is a rare class, so recall, PR-AUC and fraud F1-score are more meaningful than accuracy alone."
)