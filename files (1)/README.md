# Freight Rate Prediction

Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, absolute-error loss) predicting
log(rate per mile), validated with expanding-window chronological folds.

## Run
```bash
python -m pip install -r requirements.txt
# put the four CSVs in ./data as train_test.csv, validation.csv,
# validation_predictions_template.csv, december_chart_inputs.csv
python train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
`train.py` prints the chronological-fold and unseen-city metrics, writes `validation_predictions.csv`,
fills `data/december_chart_inputs.csv`, and saves `metrics.json`. Deterministic (fixed seed).

## Files
- `train.py` – cleaning, features, validation, final fit, predictions
- `score.py` – provided scorer (creates `scorer_results/candidate_december.png`)
- `Freight_Rate_Report.docx` – approach, validation, December chart
