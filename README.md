# Customer Spending Predictor

A Streamlit app that estimates customer spending from demographic and purchasing
details. It also includes a what-if simulator, a calibrated prediction interval,
and a page for evaluating model performance on held-out data.

## Features

- Estimate spending using age, income, purchase frequency, gender, education,
  and country.
- Explore how changing income or purchase frequency changes the model estimate.
- View a 90% prediction interval calibrated on a separate calibration set.
- Review MAE, RMSE, R-squared, and a predicted-versus-actual plot for a separate
  held-out test set.
- Compare estimates with summary statistics and charts from the dataset.

## Run locally

The commands below use PowerShell on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

If PowerShell prevents environment activation, run the app with the virtual
environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Project files

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Main Streamlit app, prediction workflow, simulator, and evaluation view |
| `spending_prediction.py` | Legacy alternate app; `streamlit_app.py` is the documented entry point |
| `customer_data.csv` | Dataset used by the app and evaluation |
| `xgboost_model.pkl` | Saved XGBoost model template |
| `scaler.pkl` | Saved feature scaler template |
| `requirements.txt` | Python package dependencies |

The Streamlit app expects the CSV and both model artifact files in the project
folder. It checks that the saved artifacts match the dataset feature count,
then creates its own reproducible training, calibration, and test splits.

## Validation and prediction interval

The app uses a fixed random seed to split the dataset into three non-overlapping
sets:

- 60% to fit the model and scaler.
- 20% to calibrate the 90% prediction interval from absolute prediction errors.
- 20% to report held-out performance metrics.

MAE and RMSE are measured in the dataset's spending units; lower values indicate
smaller errors. R-squared describes how much of the variation in the held-out
targets is explained by the model. The observed calibration coverage is shown
in the app. It is measured on this dataset and does not guarantee future
coverage or establish a confidence score.

The what-if simulator shows how the model responds to changed input values. It
does not prove that changing income or purchase frequency causes spending to
change.

## Data privacy

The `name` values in the included CSV have been replaced with generated customer
IDs. Other fields, including demographic values, remain. Before making this
repository public, confirm that you have permission to publish the underlying
data and that the remaining fields cannot identify real people. If the records
are real or sensitive, do not publish the CSV or artifacts trained from it.

## Model artifact safety

The `.pkl` files are included so the project works as provided. Only load pickle
artifacts from sources you trust; loading a malicious pickle can execute code.

## License

No license is included. Add a license only after deciding what permissions you
want to grant others for the code and data.
