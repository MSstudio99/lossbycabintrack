# EDC Cabin Loss Dashboard - Streamlit

This is a Streamlit upgrade of the notebook-based cabin loss dashboard.

## Data design

- Keep historical CSV files in the GitHub repository:
  - `data/2024/BMC.csv`, `data/2024/BTB.csv`, ...
  - `data/2025/BMC.csv`, `data/2025/BTB.csv`, ...
- Do **not** keep 2026 files in the repository. Upload them through the app using the multiple CSV uploader.
- Required province codes:
  `BMC`, `BTB`, `KP`, `KPC`, `KPS`, `KT`, `MDK`, `PV`, `RTK`, `SHV`, `SR`, `ST`, `SVR`, `TBK`, `TK`.

The app infers the province from the uploaded filename. Best filename format:

```text
BMC.csv
BTB.csv
KP.csv
...
```

It also accepts names like `BMC_2026.csv`, but do not rely on messy filenames for operational work.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy from GitHub

1. Create a private GitHub repository.
2. Upload `app.py`, `requirements.txt`, this README, and the `data/2024` and `data/2025` CSV files.
3. Deploy using Streamlit Community Cloud or your internal server.

## Important operational warning

If the historical CSVs contain customer-level or commercially sensitive data, do **not** push the repository publicly. Use a private repository or an internal deployment. A public GitHub repository is not acceptable for utility metering data.
