# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras) ![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📌 Problem Statement

A telecom company wants to proactively identify customers who are likely to cancel their service (**churn**) so that retention teams can intervene before the customer leaves.

This project builds and analyses a **Feed-Forward Neural Network (FFNN)** to classify each customer as:
- `0` → **Retained** (will stay)
- `1` → **Churned** (will cancel)

The focus is not just on achieving a good model, but on demonstrating **how neural networks learn** — through forward pass, loss calculation, backpropagation, and weight updates.

---

## 📂 Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md                         ← You are here
├── notebook.ipynb                    ← Main notebook (all 6 tasks)
├── requirements.txt                  ← Python dependencies
├── customer_churn_nn.csv             ← Dataset
│
└── results/
    ├── task1_eda_overview.png        ← EDA: distributions + target
    ├── task1_correlation_heatmap.png ← Feature correlation matrix
    ├── task1_categorical_analysis.png← Churn rate by category
    ├── task2_preprocessing.png       ← Scaling + split verification
    ├── task4_training_history.png    ← Loss + accuracy curves (baseline)
    ├── evaluation_outputs.png        ← Confusion matrix + ROC + probability dist
    ├── model_comparison_table.png    ← Hyperparameter comparison (PNG)
    ├── model_comparison_table.csv    ← Hyperparameter comparison (CSV)
    ├── task5_experiment_comparison.png ← Bar chart comparison across 7 configs
    └── task5_all_training_curves.png ← Per-config training curves
```

---

## 📊 Dataset Description

**File:** `customer_churn_nn.csv`  
**Rows:** 2,000 customers  
**Columns:** 17 (1 ID, 4 categorical, 11 numerical/binary, 1 target)

| Column | Type | Description |
|---|---|---|
| `customer_id` | ID | Unique identifier (dropped before training) |
| `region` | Categorical | South / West / Central / East |
| `plan_type` | Categorical | Basic / Standard / Premium |
| `contract_type` | Categorical | Month-to-month / Annual |
| `payment_method` | Categorical | Debit Card / Credit Card / Wallet / Net Banking |
| `tenure_months` | Numerical | Months the customer has been subscribed |
| `monthly_charges_inr` | Numerical | Monthly bill amount in INR |
| `avg_login_days_per_month` | Numerical | Average app/portal logins per month |
| `support_tickets_last_90_days` | Numerical | Support tickets filed in last 90 days |
| `payment_delay_days` | Numerical | Average payment delay in days |
| `data_usage_gb` | Numerical | Data consumed per month (GB) |
| `satisfaction_score` | Numerical | Survey satisfaction (1–10) |
| `last_complaint_days_ago` | Numerical | Days since last complaint |
| `discount_percent` | Numerical | Active discount percentage |
| `autopay_enabled` | Binary | 1 = AutoPay active |
| `referral_count` | Numerical | Number of referrals made |
| `churn` | **Target** | **0 = Retained, 1 = Churned** |

---

## 🧠 Neural Network Architecture

```
Input Layer          (29 features after encoding)
       │
Dense(128) ──► BatchNormalization ──► ReLU ──► Dropout(0.3)
       │
Dense(64)  ──► BatchNormalization ──► ReLU ──► Dropout(0.3)
       │
Dense(32)  ──► BatchNormalization ──► ReLU ──► Dropout(0.3)
       │
Dense(1, Sigmoid)   → Output: P(churn) ∈ (0, 1)
```

| Component | Choice | Reason |
|---|---|---|
| **Activation (hidden)** | ReLU | Fast, avoids vanishing gradient |
| **Activation (output)** | Sigmoid | Maps to [0,1] probability |
| **Loss function** | Binary Cross-Entropy | Standard for binary classification |
| **Optimiser** | Adam (lr=0.001) | Adaptive learning rate, fast convergence |
| **Regularisation** | Dropout + BatchNorm | Prevent overfitting |

---

## 🔬 Hyperparameter Experiments

Seven configurations were tested. Key changes:

| # | Configuration | Change | Test Acc | ROC-AUC | F1 |
|---|---|---|---|---|---|
| C1 | Baseline | Reference | ~0.872 | ~0.912 | ~0.854 |
| C2 | Shallow | 1 hidden layer only | ~0.843 | ~0.878 | ~0.819 |
| C3 | Wider | 256→128→64 neurons | ~0.878 | ~0.919 | ~0.861 |
| C4 | High LR | lr = 0.01 | ~0.851 | ~0.887 | ~0.831 |
| C5 | Low LR | lr = 0.0001 | ~0.832 | ~0.871 | ~0.810 |
| C6 | Batch 128 | batch_size = 128 | ~0.863 | ~0.902 | ~0.846 |
| C7 | tanh | tanh activation | ~0.868 | ~0.901 | ~0.849 |

**Winner:** Config 3 (wider network) achieves best ROC-AUC.

---

## 📈 Key Results

| Metric | Baseline Value |
|---|---|
| Test Accuracy | ~87–89% |
| ROC-AUC | ~0.91–0.92 |
| Precision | ~0.85 |
| Recall | ~0.86 |
| F1-Score | ~0.85 |

---

## 💬 Reflection Summary

| Question | Answer Summary |
|---|---|
| **Weights & Biases** | Weights scale inputs; biases shift the activation. Both are updated via backpropagation to minimise loss. |
| **Why activation?** | Without non-linear activations, a deep network collapses to a single linear function — unable to model complex patterns. |
| **LR too high** | Overshoots the loss minimum → unstable, oscillating validation loss. |
| **LR too low** | Tiny steps → very slow convergence; may underfit within epoch budget. |
| **Overfitting?** | Baseline showed good generalisation (small train-val gap). Config 2 (shallow) showed underfitting. |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebook.ipynb` and `customer_churn_nn.csv`
3. Run: **Runtime → Run all**

> 💡 Add this cell at the top for Colab:
> ```python
> !pip install tensorflow scikit-learn matplotlib seaborn -q
> ```

### Option 2 — Local

```bash
# Clone repo
git clone https://github.com/<your-username>/part-1-neural-network-analysis
cd part-1-neural-network-analysis

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebook.ipynb
```

---

## 📋 Requirements

See `requirements.txt`. Core dependencies:

```
tensorflow>=2.12.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 📚 References

- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
- TensorFlow Documentation: [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
- Scikit-learn User Guide: [scikit-learn.org/stable](https://scikit-learn.org/stable)
