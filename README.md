# AI Scale Web App

Interactive Streamlit demo for AI-assisted psychological scale development.

This project integrates two scale-generation frameworks:

- Top-down: generate and filter items using predefined dimensions.
- Bottom-up: generate a broad item pool, infer candidate dimensions, and filter representative items.

It also supports three prompt styles (conservative, standard, creative), demo datasets, quality summaries, Excel export, and optional API-based real generation.

## Online deployment

The recommended host is Streamlit Community Cloud. Use `app.py` as the entry point.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Security

Do not commit real API keys. If real API generation is enabled online, store credentials in Streamlit Community Cloud Secrets.
