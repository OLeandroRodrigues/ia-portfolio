# 📝 Google Reviews NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains an **experimental NLP pipeline** to analyze and classify **Google Reviews**.
The project is organized into **two layers**:

1. **Comment Classification**
   - 🎭 **Sentiment analysis** (`positive | neutral | negative`)
   - 🏷️ **Topic classification** (e.g., service, price, quality)
   - 🚨 **Spam/legitimacy detection** (`suspect | legitimate`)

2. **Author Linking Experiments**
   - 🔍 Detects possible **same-author patterns across different accounts**
   - ✍️ Based on **writing fingerprint + semantic embeddings + clustering**

---

## 🎯 Objectives
- ✅ Train and evaluate **classification models** using TensorFlow.
- ✅ Extract **stylometry features** and multilingual embeddings.
- ✅ Build similarity graphs to detect **repeated authorship**.
- ✅ Provide explainable outputs for research and audit.

---
### 🚀 Project Pipeline

1. **Preprocessing (DONE)**

-  Implemented in src/data/preprocess.py to clean and normalize raw comments.
-  Output: data/processed/data-google-reviews_clean.csv
-  Area: NLP (text cleaning & normalization)

2. **Embedding Extraction (WORK IN PROGRESS)**

-  Implemented in src/features/embeddings.py (e.g., using TF-Hub multilingual USE).
-  Output: artifacts/features/embeddings.npy
-  Area: NLP / Representation Learning (vectorization of text)

3. **Classifier Training (TO DO)**

-  Implemented in src/models/classifier.py to train a sentiment/spam detection model.
-  Output: artifacts/models/
-  Area: NLP + Machine Learning / Neural Networks

4. **Author Linking (TO DO)**

-  Implemented in src/linking/similarity.py + src/linking/clustering.py to cluster possible repeated authors.
-  Output: artifacts/predictions/author_clusters.csv
-  Area: NLP + Unsupervised Learning (clustering)

5. **Evaluation (TO DO)**

-  Implemented in src/eval/ for classification and linking metrics.
-  Area: General Machine Learning (evaluation metrics)

6. **Automated Tests (DONE)**

-  Unit tests located in tests/ for each module.
-  Area: Software Engineering / Best Practices

---

## 📂 Project Structure

```text
google-reviews-nlp/
├── README.md
├── requirements.txt
│
├── artifacts/                # generated outputs
│   ├── features/
│   ├── models/
│   └── predictions/
│
├── data/                     # datasets
│   ├── processed/
│   └── raw/
│       └── clean-data-google-reviews.csv
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_train_classifier.ipynb
│   ├── 03_linking_experiments.ipynb
│   └── .ipynb_checkpoints/
│
├── src/                      # source code
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── heuristics_labeling.py
│   │   └── preprocess.py
│   │
│   ├── eval/
│   │   ├── metrics_classification.py
│   │   └── metrics_linking.py
│   │
│   ├── features/
│   │   ├── embeddings_tfhub.py
│   │   └── stylometry.py
│   │
│   ├── link_author/
│   │   ├── build_similarity_graph.py
│   │   └── cluster_dbscan.py
│   │
│   └── models/
│       ├── classifier_tf.py
│       └── siamese_tf.py
│
└── tests/                    # unit tests
    ├── src
         ├── data
             └── test_preprocess.py
```

---

## ⚙️ Tech Stack
- 🐍 **Python 3.10+**
- 🔶 **TensorFlow / Keras** – model training
- 🌍 **TensorFlow Hub** – multilingual embeddings (USE / LaBSE)
- 📊 **scikit-learn** – clustering (DBSCAN, HDBSCAN) & metrics
- 📚 **spaCy (pt)** – optional preprocessing for Portuguese
- 📈 **pandas / matplotlib / seaborn** – analysis & visualization

---

## 🔬 Workflow

1. **Exploratory Data Analysis** (`01_eda.ipynb`)
   - Inspect raw reviews, distributions & anomalies
   - Apply basic cleaning

2. **Classification** (`02_train_classifier.ipynb`)
   - Train sentiment / topic / spam classifiers
   - Evaluate with accuracy, F1-score & confusion matrix

3. **Author Linking** (`03_linking_experiments.ipynb`)
   - Generate embeddings + stylometry features
   - Build similarity graph & cluster comments
   - Identify **candidate same-author groups**

---

## 📊 Outputs
- 📑 **Classification reports** (JSON, PNG plots)
- 🤖 **Saved TensorFlow models** (`.h5`, `.pkl`)
- 🔗 **Similarity graphs** of comments
- 👥 **Author cluster assignments** (`author_clusters.csv`)

---

## 🧪 Running the Unit Tests

Follow the steps below to execute the test suite for src/data/preprocess.py.

📦 Prerequisites

Python 3.10+

pip (virtual environment recommended)

⚙️ Setup & Install
# from the repository root
python -m venv .venv


Activate the venv

🪟 Windows

.venv\Scripts\activate


🍎 macOS / 🐧 Linux

source .venv/bin/activate


Install deps

pip install -r requirements.txt


▶️ Run the Tests
# run the whole suite
python -m pytest -q


🛠️ Troubleshooting

🐍 ModuleNotFoundError: No module named 'src'

Make sure you run from the repo root and use the module form:

python -m pytest -q

---

## ⚠️ Disclaimer
This project is intended for **research and educational purposes only**.
Author linking is **probabilistic** and should be interpreted as **candidate same-author groups**, not definitive proof.
All datasets used are either **publicly available** or **synthetic**.

---

## 🚀 Next Steps
- Improve weak labeling for "suspect vs legitimate" reviews.
- Train a Siamese/contrastive model for authorship verification.
- Add temporal & behavioral features (time, frequency).
- Create an interactive dashboard for results.
