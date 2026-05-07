# Personality Predictor — Introvert vs Extrovert

> An end-to-end machine learning pipeline that predicts whether a person is an **Introvert** or **Extrovert** based on 7 behavioral features, deployed as a production REST API on Render.

<br>

<img width="1902" height="774" alt="image" src="https://github.com/user-attachments/assets/bf842e33-0869-42b7-8b3b-4b5561fa0e1a" />
<img width="1919" height="962" alt="image" src="https://github.com/user-attachments/assets/f490ceac-4f44-4cd3-8eb1-d28ea277bad7" />
<img width="1909" height="959" alt="image" src="https://github.com/user-attachments/assets/7138457e-5541-4226-959b-5441de96ba96" />
<img width="1919" height="966" alt="image" src="https://github.com/user-attachments/assets/28e2b541-50f0-4971-b1f2-a98f933d94d7" />


---

## 📚 Table of Contents

- [Project Overview](#-project-overview)
- [Live API](#-live-api)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [File Structure](#-file-structure)
- [ML Pipeline](#-ml-pipeline)
  - [Step 1 — EDA](#step-1--exploratory-data-analysis-eda)
  - [Step 2 — Preprocessing](#step-2--preprocessing--feature-engineering)
  - [Step 3 — Train/Test Split](#step-3--traintest-split)
  - [Step 4 — Model Training](#step-4--model-training)
  - [Step 5 — Evaluation & Selection](#step-5--model-evaluation--selection)
  - [Step 6 — Model Saving](#step-6--model-saving--deployment)
- [Model Evaluation and Selection](#-model-evaluation-and-selection)
- [API Usage](#-api-usage)
- [Setup & Installation](#-setup--installation)

---

## 🔍 Project Overview

This project builds a **binary classification system** that predicts human personality type (Introvert / Extrovert) from behavioral data. The full workflow covers data exploration, preprocessing, multi-model training with overfitting detection, metric-based evaluation, and deployment as a live REST API.

**Key engineering decisions:**
- Three models trained and compared — Logistic Regression (baseline), Random Forest, and XGBoost
- Training vs Testing accuracy tracked on every model to detect overfit/underfit
- Random Forest selected for deployment based on generalization performance and production maintainability
- Model artifact saved with preprocessing state embedded for consistent inference

---

## 🚀 Live API

| Resource | URL |
|---|---|
| **Swagger UI (Docs)** | https://personality-predictor-api.onrender.com/docs |
| **Base URL** | https://personality-predictor-api.onrender.com |
| **Health Check** | `GET /health` |
| **Prediction** | `POST /predict` |

> ⚠️ Hosted on Render free tier — first request may take ~50 seconds to wake the instance.

---

## 🛠 Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **Language** | Python 3.11 | Stability + full ML ecosystem support |
| **Data** | Pandas, NumPy | Tabular data manipulation and numerical ops |
| **Visualization** | Matplotlib, Seaborn | EDA plots and model evaluation charts |
| **ML Models** | Scikit-learn, XGBoost | Industry-standard classification libraries |
| **Model Persistence** | Joblib | Efficient serialization of sklearn objects |
| **API Framework** | FastAPI | Async, auto-documented, production-ready REST APIs |
| **Deployment** | Render | Zero-config cloud deployment with GitHub integration |
| **API Testing** | Postman, Swagger UI | Manual and documented endpoint testing |
| **Notebook** | Jupyter Notebook | Reproducible ML experimentation |

---

## 📊 Dataset

**Source:** [personality_dataset.csv — Google Drive](https://drive.google.com/file/d/1GNx9VkdrFP5Tz3XsuBzscXbiYJeamBfS/view)

| Property | Value |
|---|---|
| **Task** | Binary Classification |
| **Target** | `Personality` — Introvert / Extrovert |
| **Features** | 7 behavioral features |
| **Class Balance** | 51.4% Extrovert / 48.6% Introvert ✅ Nearly balanced |

**Feature descriptions:**

| Feature | Type | Description |
|---|---|---|
| `Time_spent_Alone` | Numeric | Hours per day spent alone |
| `Social_event_attendance` | Numeric | Frequency of attending social events |
| `Going_outside` | Numeric | How often the person goes outside |
| `Friends_circle_size` | Numeric | Number of close friends |
| `Post_frequency` | Numeric | Social media posting frequency |
| `Stage_fear` | Categorical (Yes/No) | Whether the person has stage fear |
| `Drained_after_socializing` | Categorical (Yes/No) | Whether socializing drains energy |

> The dataset is **nearly balanced** (51.4% vs 48.6%), so no SMOTE or class weighting was required.

---

## 📁 File Structure

```
personality-predictor/
├── api/
│   ├── main.py
│   └── model/                  
│       └── personality_model.joblib
├── notebooks/
│       └── personality_predictor.ipynb
│       └── eda_correlation.png
│       └── eda_overview_1.png
│       └── eda_overview_2.png
│       └── model_evaluation.png
│       └── xgb_evaluation.png
├── venv/
├── requirements.txt
├── Procfile
├── render.yaml           
└── README.md
```

---

## 🔬 ML Pipeline

### Step 1 — Exploratory Data Analysis (EDA)

Three EDA plots were generated before any modelling:

**Plot 1 — Class Distribution & Missing Values**

<img width="1416" height="468" alt="image" src="https://github.com/user-attachments/assets/6ba960ad-d5eb-4869-ab9d-65409d32531e" />

- Checked class balance **first** — imbalanced data requires SMOTE or class weighting before training
- Dataset is nearly balanced (51.4% Extrovert / 48.6% Introvert) → no resampling needed
- Missing values detected in several columns → median/mode imputation applied in preprocessing

**Plot 2 — Feature Distributions by Personality**

<img width="1612" height="896" alt="image" src="https://github.com/user-attachments/assets/60d30a19-6085-4ed1-a3c9-658923677604" />

- Histograms split by personality confirm which features are strong predictors
- `Time_spent_Alone` shows the clearest separation — introverts peak at high values, extroverts at low
- `Stage_fear` and `Drained_after_socializing` show strong categorical signal
- Overlapping distributions (e.g. `Post_frequency`) indicate weaker standalone predictors

**Plot 3 — Correlation Heatmap**

<img width="929" height="811" alt="image" src="https://github.com/user-attachments/assets/3d0778af-7e6a-4a11-a43e-46c434b02cbc" />

- Identifies features most predictive of the target
- Detects multicollinearity between features — Random Forest handles this natively, so no feature removal was needed

---

### Step 2 — Preprocessing & Feature Engineering

```
Raw Data  →  Encode (Yes/No → 1/0)  →  Impute (median/mode)  →  Encode target  →  Model-ready
```

**Why median imputation?**
Median is robust to outliers unlike mean. For binary Yes/No columns, mode imputation preserves the majority class distribution.

**Why encode target as Introvert=1, Extrovert=0?**
Binary integer targets are required by scikit-learn classifiers. The label map is stored inside the model artifact so the API can decode predictions back to readable strings.

**The fill values are saved inside the model artifact** — this ensures the exact same imputation logic is applied at inference time as during training, preventing train-serve skew.

---

### Step 3 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80/20 split** — standard for medium-sized datasets
- **`stratify=y`** — preserves the original class ratio in both train and test sets, ensuring fair evaluation
- **`random_state=42`** — reproducible splits across runs

---

### Step 4 — Model Training

Three models were trained with deliberate design choices:

**Model 1 — Random Forest (Primary)**
```python
RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42)
```
- Ensemble of 200 decision trees — majority vote reduces variance
- `min_samples_leaf=2` prevents individual trees from overfitting on single data points
- No feature scaling needed — tree-based models are scale-invariant

**Model 2 — Logistic Regression (Baseline)**
```python
Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
```
- Used as the **baseline** — a simple linear model sets the floor for what "acceptable" accuracy is
- `StandardScaler` required — logistic regression is sensitive to feature magnitudes
- Wrapped in a `Pipeline` to ensure the scaler is fitted only on training data (no data leakage)

**Model 3 — XGBoost (Challenger)**
```python
XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric='logloss')
```
- Gradient boosting — each tree corrects errors of the previous one
- Built-in L1/L2 regularization reduces overfitting
- `max_depth=4` constrains tree complexity to prevent memorization

**Overfitting detection on every model:**
```
Training Accuracy vs Testing Accuracy gap > 5%  →  ⚠️ Overfit
Testing Accuracy < 75%                          →  ⚠️ Underfit
Gap within 0–5%                                 →  ✅ Good fit
```

---

### Step 5 — Model Evaluation & Selection

<img width="1917" height="557" alt="image" src="https://github.com/user-attachments/assets/a839696c-0470-4a41-ae5a-f02f4eb0b4a1" />

<img width="1538" height="586" alt="image" src="https://github.com/user-attachments/assets/c59ff05d-ed40-4f89-acf6-db41a17fb22e" />

#### Complete Metrics Comparison

| Model | Train Acc | Test Acc | Gap | CV Mean (5-fold) | Precision | Recall | F1 Score | Fit Status |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 93.19% | 90.52% | 2.67% | — | 89.27% | 91.49% | 90.37% | ✅ Good |
| **Random Forest** | **94.35%** | **91.55%** | **2.80%** | **93.03% ± 1.58%** | **89.23%** | **93.97%** | **91.54%** | ✅ **Good** |
| XGBoost | 94.31% | 91.55% | 2.76% | 92.72% ± 1.22% | 89.49% | 93.62% | 91.51% | ✅ Good |


#### Why Random Forest was selected for deployment

Both Random Forest and XGBoost achieved identical test accuracy (91.55%). The selection was made on the following grounds:

1. **Stronger cross-validation** — RF CV mean (93.03%) outperforms XGBoost (92.72%), indicating better generalization across unseen data folds
2. **More balanced feature importance** — RF distributes importance across features more evenly than XGBoost, suggesting better robustness and interpretability
3. **Lower operational complexity** — RF requires no learning rate tuning, has fewer hyperparameters, and is simpler to maintain and retrain in production
4. **Efficient inference** — prediction latency is deterministic and fast for a 200-tree ensemble on 7 features


---

### Step 6 — Model Saving & Deployment

The model is saved as a single Joblib artifact containing everything needed for inference:

```python
model_artifact = {
    'model'           : rf_model,
    'feature_columns' : feature_cols,
    'label_map'       : {0: 'Extrovert', 1: 'Introvert'},
    'fill_values'     : fill_values,       # exact imputation values from training
    'yes_no_cols'     : yes_no_cols,       # columns requiring Yes/No → 1/0 encoding
    'numeric_cols'    : numeric_cols,
    'model_accuracy'  : round(rf_acc, 4),
    'precision'       : round(precision_score(y_test, rf_preds), 4),
    'recall'          : round(recall_score(y_test, rf_preds), 4),
    'f1_score'        : round(f1_score(y_test, rf_preds), 4),
}
```

**Why embed preprocessing inside the artifact?**
At inference time, the API receives raw user input. By storing `fill_values` and column mappings inside the artifact, the API applies the exact same transformations as training — eliminating train-serve skew, which is a common production ML bug.

---

## 📈 Model Evaluation and Selection

Three machine learning models were evaluated for personality classification:

| Model | Test Accuracy | Cross-Validation Mean |
|---|---|---|
| Logistic Regression | 90.52% | — |
| Random Forest | 91.55% | 93.03% |
| XGBoost | 91.55% | 92.72% |

Both Random Forest and XGBoost achieved identical test accuracy. However, **Random Forest demonstrated slightly stronger cross-validation performance** and provided a simpler, more maintainable deployment solution.

Random Forest was selected for deployment because it offers strong generalization performance, lower implementation complexity, easier maintainability, efficient inference, and simpler operational management.

> Although XGBoost achieved similar performance, **Random Forest showed a more balanced feature importance distribution**, which may indicate better robustness and interpretability — a meaningful advantage in a production classification system.


---

## 🌐 API Usage

### POST `/predict`

**Request body:**
```json
{
  "Time_spent_Alone": 7,
  "Stage_fear": "Yes",
  "Social_event_attendance": 1,
  "Going_outside": 2,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3,
  "Post_frequency": 1
}
```

**Response:**
```json
{
  "prediction": "Introvert",
  "confidence": 0.9552
}
```

**cURL example:**
```bash
curl -X POST 'https://personality-predictor-api.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "Time_spent_Alone": 7,
    "Stage_fear": "Yes",
    "Social_event_attendance": 1,
    "Going_outside": 2,
    "Drained_after_socializing": "Yes",
    "Friends_circle_size": 3,
    "Post_frequency": 1
  }'
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Chamindu77/Personality-Predictor.git
cd Personality-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r api/requirements.txt
```

Or for notebook use:
```bash
pip install scikit-learn pandas matplotlib seaborn joblib xgboost jupyter
```

### 4. Download the dataset

Download `personality_dataset.csv` from [Google Drive](https://drive.google.com/file/d/1GNx9VkdrFP5Tz3XsuBzscXbiYJeamBfS/view) and place it in the project root.

### 5. Run the notebook

```bash
jupyter notebook notebooks/personality_predictor.ipynb
```

### 6. Run the API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

---

<div align="center">
</div>
