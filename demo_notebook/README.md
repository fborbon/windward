# Wind Farm Production Forecasting - Demo Notebook

`wind_farm_production_forecasting.ipynb` is a self-contained end-to-end case study (problem
framing, EDA, feature engineering, baseline-vs-model comparison, results, production pipeline
notes, drift monitoring, insights) built for a technical interview presentation. It reuses the
real Windward codebase (`forecasting/`, `analysis/`, `data_sources/`) rather than reimplementing
the logic - see the notebook's own §7 for the module map.

The notebook already ships with real, baked-in outputs (plots, tables), so it can be opened and
read without re-running anything. It can also be re-run live (needs internet access, for the
Open-Meteo weather API).

`wind_farm_production_forecasting_ES.ipynb` is the same notebook fully translated to Spanish
(narrative, plot labels, print/display output, code comments) - code itself (variable/function
names) is left in English, standard practice. Both notebooks are built from independent scripts
and executed separately, so their exact numbers can differ slightly between runs (the synthetic
day-ahead price series isn't seeded), but the story is identical.

`wind_prediction_techniques_showcase.ipynb` is a different case study built from the
`wind_prediction/` module (README §14) rather than `forecasting/`: an explicative tour of ten
time-series forecasting paradigms (naive through zero-shot foundation models), their real
benefits/limitations, and their real backtested performance on Kelmarsh - two techniques (naive
baselines, Gradient Boosting) run live in the notebook, the other eight are shown as real code
with results loaded from the last `python -m wind_prediction.evaluate` run
(`dashboard/data/wind_prediction_payload.json`). It also adds sections built for a specific
Senior Data Scientist - Logistics/Forecasting vacancy this was prepared against (kept generic in
the notebook's own text, see `JD_Senior_Data_Scientist_Forecasting (1).pdf` in this folder for
the original posting): translating a forecast into an inventory/safety-stock decision (newsvendor
model, real forecast error), a capacity-allocation linear program (PuLP, real per-farm production
data), and PoC-to-production/CI-CD/Airflow/Spark/cloud sections mapped onto what's real in this
repo vs. clearly-labeled illustrative code where the requested tooling (Jenkins, Spark) isn't
installed in this environment.

## Running it

A Jupyter kernel for Windward's virtualenv is already registered on this machine as **"Windward
(.venv)"**. Open the notebook in Jupyter Lab/Notebook or VS Code and select that kernel, then
"Run All".

If opening on a different machine, from the `windward/` project root:

```bash
.venv/bin/pip install ipykernel nbformat nbclient
.venv/bin/python -m ipykernel install --user --name windward --display-name "Windward (.venv)"
jupyter lab demo_notebook/wind_farm_production_forecasting.ipynb
```

Requires the Kelmarsh SCADA data at `data/kelmarsh/` (already present in this repo).
