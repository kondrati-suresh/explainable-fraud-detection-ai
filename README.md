# Explainable Fraud Detection Pipeline (AI in Finance)
**Course:** 25ML35T Foundations of Artificial Intelligence — Micro Level Project  
**Batch:** 15 | **Seed:** `113025148045`
# Batch 15 Explainable Fraud Detection Pipeline

## Project overview

The **Batch 15 Explainable Fraud Detection Pipeline** is an educational project that demonstrates how a supervised machine-learning model can identify potentially fraudulent transactions and present its results in a local website. The website has a CyberShield-inspired dashboard design, while the project itself focuses on fraud detection and explainable model output.

The system has two data modes:

- **Synthetic demonstration mode** is available by default. It uses generated transaction-like records so the project can run without a private or downloaded dataset.
- **Public dataset mode** lets a user upload a compatible `creditcard.csv` file and view a model evaluation for that data. The public dataset has not been run as part of the recorded results in this project; the included performance numbers below are synthetic-only.

This project is intended for learning and demonstration. It is not a production fraud-prevention system and should not be used to make decisions about real transactions.

## Problem being explored

Fraud cases are rare compared with legitimate transactions. As a result, a model can achieve high overall accuracy while still missing many fraud cases. The project demonstrates this class-imbalance challenge and presents several evaluation measures so a reviewer can see more than accuracy alone.

## How the project works

1. Load the built-in synthetic data or upload a compatible labeled CSV.
2. Validate the data and separate features from the target label.
3. Split the records into training and holdout sets. If a `Time` column is present, the most recent 20% is held out; otherwise, the split is stratified.
4. Standardize numeric features and train a class-weighted logistic-regression classifier with L2 regularization.
5. Display the dataset profile, class balance, evaluation measures, and confusion matrix in the dashboard.
6. Allow a transaction to be scored and show signed feature contributions to the model's result.

The explanation shows how the model's numeric features contribute to its calculated score. It does not show causation or prove why a transaction was fraudulent.

## Website features

- Local dashboard with overview and dataset profile
- Dataset preview and legitimate/fraud class balance
- Holdout evaluation with accuracy, precision, recall, F1, and confusion matrix
- Transaction scoring with probability and feature-contribution explanation
- Public CSV upload flow for compatible data
- Light and dark appearance options

The website runs locally using Python's standard library and NumPy; it does not require Flask or a separately hosted service. Its email-suffix login is only a local access gate and does not verify mailbox ownership.

## Recorded demonstration results

The checked-in results use the synthetic generator with seed `15` and a holdout of `240` rows:

| Measure | Synthetic demonstration result |
| --- | ---: |
| Accuracy | 0.9208 |
| Precision | 0.5000 |
| Recall | 0.1053 |
| F1 score | 0.1739 |
| Confusion matrix | `[[219, 2], [17, 2]]` |

The confusion matrix rows represent actual legitimate/fraud classes, and columns represent predicted legitimate/fraud classes. These figures describe only the generated synthetic example. They are not results from the public dataset and are not evidence of real-world performance.

## Public dataset

The website accepts a CSV containing numeric feature columns and a binary `Class` column (`0` for legitimate and `1` for fraud). The documented ULB/Worldline dataset contains 284,807 anonymized transactions from two days in 2013, including 492 fraud labels. It is historical and highly imbalanced. The raw CSV is intentionally not included in this project; download it from the dataset provider if you want to try the upload feature. Dataset account and terms may apply.

- [ULB / Worldline Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [OpenML dataset metadata, ID 43627](https://www.openml.org/search?id=43627&sort=runs&type=data)

The application stores an uploaded file locally at `data/creditcard.csv`. If a `Time` column is present, it holds out the last 20% by time; otherwise, it uses a stratified 80/20 split. Training applies balanced class weights and caps majority-class training rows at 60,000. A successful upload does not establish the provenance, representativeness, or suitability of the data.

## Run the website

1. Install Python 3.10 or later and the project dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Start the site by double-clicking `start_website.bat`, or run `run_website.ps1` from PowerShell in the project folder.
3. Enter the local sign-in details shown by the site. Keep the launcher window open.
4. Open the local address printed by the launcher (normally `http://127.0.0.1:8000`).
5. To stop the server, press `Ctrl+C` in the launcher window.

To use the public dataset, download `creditcard.csv`, sign in, then use **Dataset profile** to select the file and choose **Upload and train model**.

## Run the pipeline and checks

Generate and save the synthetic demonstration metrics:

```bash
python src/fraud_pipeline.py --seed 15
```

Score an example transaction:

```bash
python src/fraud_pipeline.py --amount 420 --age 12 --velocity 6 --distance 90 --device-risk 0.8
```

Run the included checks:

```bash
python tests/test_pipeline.py
```

The metric output is written to `outputs/demo_metrics.json`. The included tests use small fixtures for the CSV upload path; they do not run the full Kaggle dataset.

## Project files

```text
src/
  fraud_pipeline.py     Synthetic pipeline and model evaluation
  real_dataset.py       CSV validation, data split, and real-data model path
  webapp.py             Local website and dashboard
 data/                  Local upload location; raw CSV is not bundled
 docs/                  Academic report in PDF and Word formats
 outputs/               Saved synthetic demonstration metrics
 tests/                 Project checks
 README.md               Project overview and run guide
 requirements.txt        Python dependencies
 start_website.bat       Website launcher
 run_website.ps1         PowerShell website launcher
```

## Limitations and responsible interpretation

- Synthetic demonstration metrics come from artificial data generated by the project and should not be generalized to real payments.
- The public CSV upload feature exists, but no full public-dataset evaluation is included in the recorded results.
- A single holdout and a fixed 0.5 threshold are not enough for production validation or calibration.
- The dataset is historical; results on it would not represent current transaction patterns.
- Feature contributions explain model arithmetic, not causality.
- Real deployment would require representative and lawfully used data, leakage checks, independent temporal validation, calibration, privacy and security review, monitoring, and human investigation.

## Academic report

The accompanying academic report follows the supplied Foundations of Artificial Intelligence micro-project template:

- `docs/Batch15_Fraud_Detection_Micro_Project_Report.pdf`
- `docs/Batch15_Fraud_Detection_Micro_Project_Report.docx`

The student identity fields, GitHub details, and authentic running-website screenshots should be completed by the student where required.


## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run master prototype: `python -m src.main`
3. Run tests: `python -m unittest discover tests/`
