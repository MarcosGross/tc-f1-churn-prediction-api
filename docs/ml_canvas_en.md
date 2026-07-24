# Machine Learning Canvas (OWNML) — Churn Prediction

Designed for: Telco Retention Program · Designed by: [Group Names Here] · Iteration: 1

> Blocks marked **(ASSUMPTION)** are draft defaults filling gaps not covered in the
> original Portuguese notes. Validate/adjust with the group before finalizing.

---

## 1. Value Proposition
A telecom operator is losing customers (churn) at an accelerated rate. Leadership
needs to identify, BEFORE cancellation, which customers are at high risk of leaving,
so the retention team can act (discount, upgrade offer, proactive contact).

**End beneficiary**: the retention/CRM team, backed by commercial leadership who
wants to reduce monthly churn rate.

## 2. Prediction Task
- **Type of task**: Supervised binary classification.
- **Entity**: Individual customer (`customerID`).
- **Possible outcomes**: `Churn` = Yes / No.
- **When outcomes are observed**: In this static dataset, churn is observed as a
  snapshot at data collection time. In a production setting, the outcome would be
  observed at contract cancellation/non-renewal events, logged by the billing/CRM
  system. *(ASSUMPTION — confirm interpretation with group)*

## 3. Data Sources
- Telco Customer Churn (IBM) — public tabular dataset.
- Source: github.com/IBM/telco-customer-churn-on-icp4d (`data/Telco-Customer-Churn.csv`).
- Volume: 7,043 records, 21 columns (to confirm exact count after real EDA).

## 4. Data Collection **(ASSUMPTION)**
- Initial set of entities/outcomes sourced from the IBM sample dataset, acting as a
  proxy for a real production extract from billing + CRM systems.
- For this academic project, the dataset is a static, one-time snapshot — no
  continuous refresh strategy is implemented.
- In a real deployment, data would be extracted periodically (e.g., daily/weekly ETL
  job from the billing database), a strategy to be documented as future work in the
  Model Card / monitoring plan (Stage 4).

## 5. Features
- **Candidate features**: tenure, contract type, payment method, contracted
  services (internet, streaming, tech support), monthly and total charges.
- **Representation** *(ASSUMPTION)*: numerical features scaled (tenure,
  MonthlyCharges, TotalCharges); categorical features one-hot or ordinal encoded
  (Contract, PaymentMethod, InternetService, etc.); binary Yes/No fields mapped to 0/1.
- **Aggregations/transformations** *(ASSUMPTION)*: dataset is already aggregated at
  customer level (one row per customer), so no cross-entity aggregation is needed.
  Candidate engineered features to explore in EDA: tenure buckets, total number of
  subscribed services.

## 6. Building Models **(ASSUMPTION)**
- **How many models**: 3 candidates compared, all within Scikit-Learn — Logistic
  Regression (baseline), Random Forest/ensemble (tree-based), and `MLPClassifier`
  (simple neural network). The best-performing one on the metrics below is chosen
  as the champion and saved with `joblib`.
- **Experiment tracking**: lightweight for this project — a comparison table
  (notebook or spreadsheet) is enough; MLflow or similar tools are optional, not
  required.
- **When to update**: out of scope for this academic project (static dataset), but
  documented as a recommendation: retrain periodically (e.g., monthly/quarterly) as
  new labeled churn data accumulates in production.
- **Time/resources available**: training must run within a standard notebook/CI
  session; all candidate models are CPU-only, no GPU required.

## 7. Offline Evaluation
- **Primary metric**: AUC-ROC (robust to class imbalance).
- **Secondary metrics**: PR-AUC (churn is typically the minority class), F1-score,
  Recall of the "Churn" class (missing a true churner is costly).
- **Minimum baseline**: Logistic Regression is the baseline model itself; Random
  Forest and MLPClassifier must outperform it to be considered as champion.
- **Validation strategy**: Stratified K-Fold cross-validation, fixed random seeds.

## 8. Decisions
- Predictions are turned into a risk score/flag per customer.
- Customers above a defined risk threshold are added to an "at-risk" list consumed
  by the CRM/retention team.
- Retention team selects the specific action (discount, upgrade offer, proactive
  call) based on customer segment — action selection itself is a manual, out-of-model
  decision for this MVP.

## 9. Making Predictions **(ASSUMPTION)**
- This project delivers a real-time synchronous API endpoint (`/predict`, FastAPI),
  as required by the Tech Challenge.
- **Frequency**: on-demand call per customer interaction; alternative production
  rollout as a daily/weekly batch job is discussed but formally decided in Stage 4
  (deploy architecture).
- **Time budget**: p95 latency < 300ms (see SLOs below), including featurization.
- **Compute resources**: single small CPU instance — MLP is lightweight enough not
  to require GPU at inference time.

## 10. Impact Simulation **(ASSUMPTION — needs real business input)**
- **Cost/gain values** (illustrative placeholders, replace with real figures if the
  group has access to them): cost of a retention offer ≈ R$50/customer; cost of
  losing a customer (lost LTV) ≈ R$1,200/customer.
- **Business cost function**: total cost = (false negatives × cost of losing a
  customer) + (false positives × cost of a wasted retention offer).
- **Data used to simulate**: historical churn labels + billing values from the
  held-out test set.
- **Deployment criteria**: the model must outperform baselines on this business cost
  function computed on the hold-out set, not just on AUC-ROC alone.
- **Fairness constraints**: monitor recall / false-positive-rate gap across
  sensitive groups (e.g., gender, SeniorCitizen); flag if disparity exceeds a defined
  threshold (e.g., 10 percentage points) — document findings in the Model Card.

## 11. Monitoring **(ASSUMPTION)**
- **SLOs**: inference latency p95 < 300ms; `/health` endpoint returning 200
  continuously; throughput sized to active customer base.
- **KPIs tracked post-deployment**: live AUC-ROC/PR-AUC/F1 as new labels arrive,
  business cost metric, data drift on key features (tenure distribution, monthly
  charges), prediction volume.
- **Review frequency**: monthly review cadence, or whenever a new labeled batch
  becomes available; alert if AUC-ROC drops more than 0.03 from baseline.

---

## Known risks & potential biases (carried over from group notes)
- Dataset may not reflect real seasonality (it's a static snapshot).
- Risk of bias from sensitive variables (e.g., gender, dependents) — document in the
  Model Card and evaluate whether to remove or handle them carefully.
- Risk of data leakage if any feature only exists after churn has already happened.
