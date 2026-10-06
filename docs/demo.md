# 3–5 Minute Demo

1. Start EventLens with `streamlit run app.py` and upload `sample_data/sample_events.csv`.
2. Point out the row/column overview, missing values, and the one intentional duplicate event.
3. In **Interactive Analysis**, show average `amount` by `country` or the monthly sum trend. The sample records $362.49 in September and no purchase amount in October.
4. In **Product Analytics**, show the funnel `view → add_to_cart → purchase`. The sample contains those exact event names.
5. Show D7 retention and explain that recent cohorts can be unavailable when the dataset does not cover the full follow-up period.
6. In **AI Analyst**, ask “What is DAU?” for a direct request. If an API key is configured, follow with “Why did revenue drop in the latest month?” and show the bounded plan and evidence.
7. Without an API key, point out the setup message; upload, EDA, interactive analysis, and product analytics remain usable.

AI responses require an OpenAI-compatible provider and can vary in their plan wording. The numeric results are computed locally from the uploaded CSV.
