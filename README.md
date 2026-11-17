# ML Network Intrusion Detection

A reproducible, offline machine-learning baseline for classifying labeled
network traffic as benign or as an attack category. The repository supports
NSL-KDD (the dataset already documented by this project) and labeled CSV flow
exports. It does **not** capture packets, monitor a live network, or block
traffic.

## Problem and objectives

Network intrusion datasets are typically imbalanced: a high overall accuracy
can hide failures to detect less frequent attack classes. This project provides
a leakage-aware training and evaluation workflow, compares several classifiers,
reports macro and weighted metrics, and packages preprocessing with the selected
model for consistent inference.

## Architecture

```text
data/raw/                  Local, ignored datasets (not committed)
src/data_loader.py         NSL-KDD and labeled CSV loading
src/preprocessing.py       Train-fitted imputing, encoding, scaling, labels
src/feature_selection.py   Name-based removal of identifiers/metadata
src/train.py               Stratified model selection and artifact creation
src/evaluate.py            Held-out metrics, classification report, figures
src/predict.py             Artifact validation and single-record inference
tests/                     Focused loading, preprocessing, and prediction tests
app.py                     Streamlit dashboard
models/                    Generated model bundle (not committed)
reports/                   Generated metrics and visualizations
```

## Dataset

The default dataset is **NSL-KDD**, because it is the dataset already named in
the repository's original data instructions. It contains connection records
derived from KDD Cup 99, with 41 traffic features, an attack label, and an
NSL-KDD difficulty field. The difficulty field is excluded from predictors.
The four traditional attack families (DoS, Probe, R2L, U2R) plus Benign are
used as the target classes.

Source and download instructions are in [`data/README.txt`](data/README.txt).
The official University of New Brunswick dataset page is
<https://www.unb.ca/cic/datasets/nsl.html>; the existing project note also
identifies a GitHub mirror. No dataset records are bundled or fabricated.
Check the source's terms and citation requirements before using or
redistributing downloaded data; a separate open-data license is not asserted
here.

To use the NSL-KDD files:

```bash
mkdir -p data/raw
# Place KDDTrain+.txt in data/raw.
# Optionally place the matching official KDDTest+.txt beside it.
```

Alternatively, place one or more headered CSV files under `data/raw/`. Each CSV
must have a target column named `Label`, `class`, `attack`, or `target`. CSV
feature columns are combined by name. When `KDDTrain+.txt` is available, it is
used for development and `KDDTest+.txt` (if present) is reserved for final
testing.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train
```

Optional arguments:

```bash
python -m src.train --data data/raw --models models --reports reports --cv 3
```

The pipeline removes exact duplicate labeled records, replaces numeric
infinities with missing values, drops identifier/metadata fields, imputes
missing values, one-hot encodes categories, and standardizes numeric columns.
Every learned transform is inside a scikit-learn pipeline, so it is fit only
within each training fold and then refit on train plus validation for the final
model. Rows are stratified into train/validation/test partitions; when the
official NSL-KDD test file is available, it is kept separate and used only for
final evaluation.

Training compares:

* Logistic Regression (balanced class weights)
* Decision Tree (balanced class weights)
* Random Forest (balanced subsample weights)
* Linear SVM (balanced class weights)
* K-Nearest Neighbors (distance-weighted neighbor voting)
* Gradient Boosting

Stratified cross-validation tunes the small documented parameter grids using
macro F1. The validation macro F1 selects the model; the test set is not used
for model selection. Class weighting is applied where supported; no synthetic
records or resampling are introduced. KNN and Gradient Boosting do not use
class weights in this implementation, so per-class and macro metrics should be
reviewed carefully.

Generated files:

* `models/best_model.joblib` — preprocessing and selected estimator together.
* `reports/model_comparison.csv` — cross-validation and validation comparisons,
  plus measured training and per-record validation inference time.
* `reports/evaluation_report.json` and `reports/evaluation_report.md` — held-out
  test results.
* `reports/figures/` — confusion matrix, one-vs-rest ROC and precision-recall
  plots when supported by the selected estimator and test labels.

No experiment results are included in this repository because the dataset is
not present. Results are generated only after training on the user's local data.
Accuracy alone is insufficient for intrusion detection: a majority benign
class can dominate accuracy while attacks are missed. Macro precision, recall,
F1, per-class results, confusion matrices, and (where defined) one-vs-rest
ROC-AUC are also reported. Macro averages weight classes equally; weighted
averages weight each class by its support.

## Detailed experiment report

With the official NSL-KDD train and test files in `data/raw/`, reproduce the
binary and five-category experiment outputs and generate the Word report with:

```bash
python -m src.ids_pipeline
python -m src.build_report
soffice --headless --convert-to pdf --outdir report \
  report/23ESKCY036_NishantKumawat_CSL-G38.docx
```

The experiment writes measured metrics, test checks, and figures under
`results/`. The report builder reads those files and refuses synthetic results
or dataset sizes outside the expected NSL-KDD range.

## Dashboard and prediction

```bash
streamlit run app.py
```

The dashboard provides project/model information, local dataset class
distribution and numeric correlations, actual validation/test reports and
figures, and a form for classifying one feature record. It requires a generated
model for prediction. The probability shown for probabilistic classifiers is a
model probability, not a calibrated operational risk score; the Linear SVM
does not expose a confidence probability.

Programmatic inference:

```python
from src.predict import load_model, predict_traffic

model = load_model("models/best_model.joblib")
result = predict_traffic(model, {"duration": 0.0, "protocol_type": "tcp"})
print(result)
```

Supply every feature from the trained schema, with the original feature names.
Missing or unexpected fields and invalid numeric inputs are reported explicitly.
Only load joblib artifacts from trusted sources because loading serialized
Python objects can execute code.

## Tests and quality checks

```bash
python -m pytest
python -m compileall -q src app.py tests
```

The tests cover CSV loading and deduplication, train-fitted preprocessing,
unseen categories, class-label normalization, artifact loading, feature order,
and invalid prediction inputs. The project requires local data to execute a
real training run; unit tests do not claim model performance.

## Limitations and future work

NSL-KDD is an older benchmark and is not representative of all modern networks
or threats. Data shift, synthetic/curated collection conditions, label quality,
class imbalance, and the absence of temporal/group-aware splitting limit
generalization claims. This is not a production security control and has not
been validated against live traffic. Potential future work includes temporal
or capture-level holdouts, calibrated probabilities, independent modern
datasets, drift monitoring, and operational false-positive analysis.

## References

* Canadian Institute for Cybersecurity, [NSL-KDD dataset page](https://www.unb.ca/cic/datasets/nsl.html).
* M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the
  KDD CUP 99 Data Set,” IEEE Symposium on Computational Intelligence for
  Security and Defense Applications, 2009.
