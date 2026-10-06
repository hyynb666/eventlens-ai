# EventLens AI

EventLens AI is a lightweight open-source analytics assistant for tabular and product event data, combining local calculations with controlled LLM tool use.

## Why EventLens AI

EventLens brings exploratory data analysis, interactive statistics, and common product metrics into a small Streamlit app. You can explore a CSV directly or ask an optional LLM to map a question to approved analytics tools. EventLens validates those requests and computes every metric locally.

## Features

### Exploratory Data Analysis

- Upload CSV files and preview rows, column types, missing values, and duplicate rows.
- Review numeric and categorical summaries and per-column profiles.

### Interactive Analytics

- Categorical counts and percentages, including missing values.
- Numeric summaries and histograms.
- Grouped count, sum, mean, median, minimum, and maximum.
- Daily, weekly, and monthly trends.
- Pearson and Spearman correlation matrices.

### Product Analytics

- Daily, weekly, and monthly active users (DAU, WAU, MAU).
- Revenue, revenue over time, ARPU, and ARPPU.
- Ordered conversion and two-to-five-step event funnels.
- Exact-day D1, D7, and D30 cohort retention.

### AI Analyst

- Map a natural-language question to one supported analysis request.
- Ask for direct metrics such as DAU, a grouped average, or a monthly trend.
- Requires an API key and an OpenAI-compatible Chat Completions provider.

### Guided Multi-Step Analysis

- For diagnostic questions, create a structured plan of up to five steps.
- Validate every step before running any analysis.
- Show each approved tool's evidence and a deterministic synthesis.
- Compare the latest two observed time periods when a trend has enough data.

### Safety

- Data analysis and numerical calculations run locally in EventLens.
- The LLM receives column metadata and the question, not dataset rows.
- Model output can only select from a fixed registry of validated EventLens tools.

## How It Works

```text
User question
    → LLM interpretation
    → structured request or plan
    → full validation
    → approved local analytics tools
    → verified evidence
    → deterministic answer
```

The LLM interprets the question and chooses parameters; it does not calculate metrics. EventLens validates the selected columns and values, calls its existing analytics functions, and builds the response from their results.

## Quick Start

Requires Python 3.11 or newer. Clone the repository, open Windows PowerShell in its project directory, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
```

If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. You can also omit activation and run `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` followed by `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Upload `sample_data/sample_events.csv` to try the app. All non-AI analysis features work without provider configuration.

## AI Setup

Natural-language analysis is optional. Set environment variables in the same PowerShell window before launching Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-4.1-mini"
# Optional for an OpenAI-compatible provider:
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
# Optional: choose a lower planning limit from 1 to 5 (default 5):
$env:EVENTLENS_MAX_ANALYSIS_STEPS = "3"
streamlit run app.py
```

`OPENAI_MODEL` defaults to `gpt-4.1-mini`; `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`; the guided workflow defaults to five steps and never allows more than five. Do not put real credentials in source control. If no API key is configured, the app explains how to enable AI while leaving EDA, interactive analytics, and product analytics available.

## Example Questions

With `sample_data/sample_events.csv`, try:

- “What is average amount by country?”
- “Show monthly revenue trend.”
- “What is DAU?”
- “Show conversion from view to purchase.”
- “Show a funnel from view to add_to_cart to purchase.”
- “What is D7 retention?”
- “Why did revenue drop in the latest month?”
- “What changed in user activity recently?”

The sample's actual event values are `view`, `add_to_cart`, and `purchase`. Events cover September and October 2026; September records $362.49 in purchase amounts, while no purchase amount is recorded in October. The trend therefore shows no recorded October revenue, rather than inferring why there were no purchases. User activity also falls from six distinct users in September to two in October. The sample includes missing values and a duplicate event so the upload and quality summaries have something to show.

## Sample Dataset

`sample_data/sample_events.csv` is a small e-commerce event dataset with user IDs, timestamps, event types, product IDs, purchase amounts, countries, and devices. It demonstrates categorical and numeric columns, missing purchase amounts and device values, ordered conversion/funnel events, repeat activity for retention, multiple months for trend comparisons, and one intentional duplicate row.

## Testing

Install the development extra as shown above, then run:

```powershell
python -m pytest
```

The tests use local data and fake LLM responses; they do not require a live provider call or API key.

## Security Model

- No `eval`, `exec`, arbitrary generated Python, or generated SQL is used.
- The model cannot run shell commands, access files, or choose network requests.
- Single-step requests and every step in a multi-step plan are validated against the dataset schema.
- Multi-step plans use a fixed tool registry and are validated in full before execution.
- Only compact verified results are used for final synthesis; the full dataset is not resent to the LLM.

## Limitations

- AI interpretation requires an external OpenAI-compatible provider and may return malformed or unsuitable requests.
- Available analyses are limited to the implemented EventLens tools.
- Guided analysis is limited to five steps and does not forecast or perform causal inference.
- Associations in the data are not proof that a segment caused an outcome.
- Results depend on the quality, coverage, and meaning of the uploaded columns.

## Roadmap

- Controlled period-over-period segment comparisons.
- Exportable analysis reports.
- Additional data sources such as SQL databases.
- More charting options and statistical tests.

These are possible future directions and are not implemented in this release.
