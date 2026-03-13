# appStarter.py

"""
A starter template for a Streamlit Superstore analytics dashboard. 
It assumes use the Superstore CSV (often named
Superstore.csv) with columns such as:
Order Date, Region, Category, Sub-Category, Sales, Profit, Quantity, Segment
Dataset can be found online, e.g. via Kaggle at 
https://www.kaggle.com/datasets/konstantinognev/sample-superstorecsv/data
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import BytesIO
from openai import OpenAI
import base64
import json

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# 
 
MODEL_NAME = "gpt-5.4"
# MODEL_NAME= "gpt-5-chat-latest" 

# Start with a lower token limit to reduce cost and latency, and retry with more if we see truncation.
# 900 is not enough; Reasoning consume many tokens
BASE_MAX_OUTPUT_TOKENS = 1400  

@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

 
# This function converts a Matplotlib figure (fig) into a 
# PNG image encoded as a “data URL” 
# (a string you can embed directly into HTML/CSS/Markdown 
# without saving a file).

 
def figure_to_data_url(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_monthly_summary(monthly_df: pd.DataFrame, metric_name: str) -> str:
    summary_lines = []
    for row in monthly_df[["Month", metric_name]].itertuples(index=False):
        summary_lines.append(f"- {row[0]}: {row[1]:,.2f}")
    return "\n".join(summary_lines)


def extract_response_text(response) -> str:
    direct_text = getattr(response, "output_text", "")
    if direct_text and direct_text.strip():
        return direct_text.strip()

    collected_chunks = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text_value = getattr(content_item, "text", None)
            if text_value:
                collected_chunks.append(text_value)

    if collected_chunks:
        return "\n\n".join(chunk.strip() for chunk in collected_chunks if chunk and chunk.strip())

    return ""


def analyze_chart_with_gpt(
    image_data_url: str,
    monthly_df: pd.DataFrame,
    metric_name: str,
    total_sales_value: float,
    total_profit_value: float,
    profit_margin_value: float,
) -> str:
    prompt = f"""
You are analyzing a business dashboard line chart for a Superstore dataset.

The user needs three things in this order:
1. A plain-English explanation of the chart.
2. The most important business insights.
3. A section titled "What the user should do first" with the single highest-priority next action.

Use clean markdown with short headings and concise bullets where useful.
Be specific and actionable. Do not mention model limitations unless necessary.

Context:
- Metric shown: {metric_name}
- Total Sales in current filtered view: ${total_sales_value:,.2f}
- Total Profit in current filtered view: ${total_profit_value:,.2f}
- Profit Margin in current filtered view: {profit_margin_value * 100:.2f}%

Monthly values:
{build_monthly_summary(monthly_df, metric_name)}
""".strip()

    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }
    ]

    # First pass: keep reasoning low so the model emits user-visible text quickly.
    response = get_openai_client().responses.create(
        model=MODEL_NAME,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
        max_output_tokens=BASE_MAX_OUTPUT_TOKENS, # Start with a lower token limit to reduce cost and latency, and retry with more if we see truncation.
        input=request_input,
    )
    markdown_text = extract_response_text(response)
    if markdown_text:
        return markdown_text

    incomplete = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None

    # Retry automatically if the first response was truncated at max output tokens.
    if incomplete_reason == "max_output_tokens":
        response_retry = get_openai_client().responses.create(
            model=MODEL_NAME,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
            max_output_tokens=BASE_MAX_OUTPUT_TOKENS * 2,   # 900 was not enough in some cases, so we increase for the retry
            input=request_input,
        )
        markdown_text = extract_response_text(response_retry)
        if markdown_text:
            return markdown_text

        response = response_retry

    try:
        response_payload = response.model_dump_json(indent=2)
    except Exception:
        response_payload = str(response)

    raise ValueError(
        "The model returned a response, but no displayable text was found in the payload.\n"
        f"Response payload:\n{response_payload}"
    )


def render_copy_button(markdown_text: str) -> None:
    button_id = f"copy-md-{abs(hash(markdown_text))}"
    text_json = json.dumps(markdown_text)
    components.html(
        f"""
        <div>
            <button id="{button_id}" style="padding: 0.55rem 0.9rem; border-radius: 0.5rem; border: 1px solid #d0d7de; background: white; cursor: pointer;">
                Copy markdown
            </button>
            <span id="{button_id}-status" style="margin-left: 0.75rem; font-family: sans-serif;"></span>
        </div>
        <script>
            const button = document.getElementById({json.dumps(button_id)});
            const status = document.getElementById({json.dumps(button_id + '-status')});
            const markdownText = {text_json};
            button.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(markdownText);
                    status.textContent = "Copied to clipboard.";
                }} catch (error) {{
                    status.textContent = "Copy failed.";
                }}
            }});
        </script>
        """,
        height=50,
    )

# GPT-5.2 Thinking Chart Analysis
@st.dialog("AI Insight Chart Analysis", width="large")
def show_chart_analysis_dialog() -> None:
    if st.session_state.get("chart_analysis_error"):
        st.error(st.session_state["chart_analysis_error"])
    else:
        st.markdown(st.session_state["chart_analysis_markdown"])
        st.divider()
        action_left, action_right = st.columns(2)
        with action_left:
            st.download_button(
                "Download markdown",
                data=st.session_state["chart_analysis_markdown"],
                file_name=st.session_state.get("chart_analysis_file_name", "chart_analysis.md"),
                mime="text/markdown",
                use_container_width=True,
            )
        with action_right:
            render_copy_button(st.session_state["chart_analysis_markdown"])

    if st.button("Close", use_container_width=True):
        st.session_state["show_chart_analysis_dialog"] = False
        st.rerun()


st.session_state.setdefault("show_chart_analysis_dialog", False)
st.session_state.setdefault("chart_analysis_markdown", "")
st.session_state.setdefault("chart_analysis_error", "")
st.session_state.setdefault("chart_analysis_file_name", "chart_analysis.md")
st.session_state.setdefault("chart_analysis_raw_response", "")

st.title("📊 Superstore Interactive Analytics Dashboard")
st.write("Upload the Superstore CSV to explore sales and profit with filters and charts.")

# -----------------------------
# Helper: load + clean data
# -----------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    encodings = [None, "utf-8", "ISO-8859-1", "cp1252"]
    last_error = None

    for encoding in encodings:
        try:
            file.seek(0)
            if encoding is None:
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise last_error

    # Basic cleaning / parsing (adjust if your column names differ)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Ensure numeric
    for col in ["Sales", "Profit", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key fields
    key_cols = [c for c in ["Order Date", "Sales", "Profit"] if c in df.columns]
    df = df.dropna(subset=key_cols)

    # Month column
    if "Order Date" in df.columns:
        df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload Superstore CSV", type=["csv"])

if not uploaded:
    st.info("Upload your Superstore CSV to begin.")
    st.stop()

df = load_data(uploaded)

required_columns = ["Order Date", "Sales", "Profit", "Month"]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(
        "The uploaded file is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Preview
with st.expander("🔎 Preview data"):
    st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

# Date filter
min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Categorical filters
def multiselect_filter(label, colname):
    if colname not in df.columns:
        return None
    options = sorted(df[colname].dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)

regions = multiselect_filter("Region", "Region")
categories = multiselect_filter("Category", "Category")
segments = multiselect_filter("Segment", "Segment")

profit_only = st.sidebar.checkbox("Profit > 0 only", value=False)

# -----------------------------
# Apply filters
# -----------------------------
mask = (df["Order Date"].dt.date >= start_date) & (df["Order Date"].dt.date <= end_date)

if regions is not None:
    mask &= df["Region"].isin(regions)
if categories is not None:
    mask &= df["Category"].isin(categories)
if segments is not None:
    mask &= df["Segment"].isin(segments)
if profit_only and "Profit" in df.columns:
    mask &= df["Profit"] > 0

f = df.loc[mask].copy()

# Handle empty result
if f.empty:
    st.warning("No rows match your filters. Try widening the date range or selections.")
    st.stop()

# -----------------------------
# KPI row
# -----------------------------
total_sales = float(f["Sales"].sum())
total_profit = float(f["Profit"].sum())
profit_margin = (total_profit / total_sales) if total_sales != 0 else 0.0
num_orders = f["Order ID"].nunique() if "Order ID" in f.columns else len(f)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${total_sales:,.2f}")
c2.metric("Total Profit", f"${total_profit:,.2f}")
c3.metric("Profit Margin", f"{profit_margin*100:.2f}%")
c4.metric("Orders", f"{num_orders:,}")

st.divider()

# -----------------------------
# Chart controls
# -----------------------------
metric_choice = st.radio(
    "Metric to visualize",
    options=["Sales", "Profit"],
    horizontal=True,
)

# -----------------------------
# Charts: Two-Column format; Layout Control 
# -----------------------------
left, right = st.columns(2)

# 1) Line chart: metric over time (by Month)
with left:
    st.subheader(f"{metric_choice} Over Time (Monthly)")
    monthly = f.groupby("Month", as_index=False)[metric_choice].sum()
    monthly["Month_dt"] = pd.to_datetime(monthly["Month"] + "-01")
    monthly = monthly.sort_values("Month_dt")

    fig, ax = plt.subplots()
    ax.plot(monthly["Month_dt"], monthly[metric_choice], marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel(metric_choice)
    ax.tick_params(axis="x", rotation=45)
    line_chart_image_url = figure_to_data_url(fig)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    api_key_ready = "OPENAI_API_KEY" in st.secrets
    if not api_key_ready:
        st.info("Add OPENAI_API_KEY to Streamlit secrets to enable AI chart analysis.")

    if st.button(
        "Chart insights ✨ with AI",
        use_container_width=True,
        disabled=not api_key_ready,
    ):
        with st.spinner("Uploading the chart and generating analysis..."):
            try:
                analysis_markdown = analyze_chart_with_gpt(
                    image_data_url=line_chart_image_url,
                    monthly_df=monthly,
                    metric_name=metric_choice,
                    total_sales_value=total_sales,
                    total_profit_value=total_profit,
                    profit_margin_value=profit_margin,
                )
                st.session_state["chart_analysis_markdown"] = analysis_markdown
                st.session_state["chart_analysis_error"] = ""
                st.session_state["chart_analysis_raw_response"] = ""
                st.session_state["chart_analysis_file_name"] = (
                    f"{metric_choice.lower()}_monthly_chart_analysis.md"
                )
            except Exception as exc:
                st.session_state["chart_analysis_markdown"] = ""
                st.session_state["chart_analysis_error"] = (
                    "The chart analysis request failed. "
                    f"Details: {exc}"
                )
                raw_response = getattr(exc, "response", None)
                if raw_response is not None:
                    try:
                        st.session_state["chart_analysis_raw_response"] = raw_response.model_dump_json(indent=2)
                    except Exception:
                        st.session_state["chart_analysis_raw_response"] = str(raw_response)
            st.session_state["show_chart_analysis_dialog"] = True
            st.rerun()

# 2) Bar chart: metric by Category (or Region)
with right:
    st.subheader(f"{metric_choice} by Category")
    by_cat = f.groupby("Category", as_index=False)[metric_choice].sum().sort_values(metric_choice, ascending=False)

    fig, ax = plt.subplots()
    ax.bar(by_cat["Category"], by_cat[metric_choice])
    ax.set_xlabel("Category")
    ax.set_ylabel(metric_choice)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Auto-insights
# -----------------------------
st.subheader("🧠 Quick Insights")

top_region = None
if "Region" in f.columns:
    r = f.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    if not r.empty:
        top_region = r.index[0]

top_cat_profit = None
if "Category" in f.columns:
    cp = f.groupby("Category")["Profit"].sum().sort_values(ascending=False)
    if not cp.empty:
        top_cat_profit = cp.index[0]

worst_subcat = None
if "Sub-Category" in f.columns:
    sc = f.groupby("Sub-Category")["Profit"].sum().sort_values()
    if not sc.empty:
        worst_subcat = sc.index[0]

insights = []
if top_region:
    insights.append(f"Top region by sales: **{top_region}**")
if top_cat_profit:
    insights.append(f"Highest-profit category: **{top_cat_profit}**")
if worst_subcat:
    insights.append(f"Lowest-profit sub-category: **{worst_subcat}**")
insights.append(f"Profit margin in current selection: **{profit_margin*100:.2f}%**")

for item in insights:
    st.write("• " + item)

# -----------------------------
# Optional: download filtered data
# -----------------------------
st.download_button(
    "⬇️ Download filtered data as CSV",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="superstore_filtered.csv",
    mime="text/csv",
)

if st.session_state.get("show_chart_analysis_dialog"):
    show_chart_analysis_dialog()

if st.session_state.get("chart_analysis_error") and st.session_state.get("chart_analysis_raw_response"):
    with st.expander("Model response debug details"):
        st.code(st.session_state["chart_analysis_raw_response"], language="json")