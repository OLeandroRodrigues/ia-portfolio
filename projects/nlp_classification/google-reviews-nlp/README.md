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

## 🏷️ Labeling Strategy (heuristics)

The file [`reviews_labeled.csv`](data/processed/reviews_labeled.csv) is generated automatically by the script [`src/data/heuristics_labeling.py`](src/data/heuristics_labeling.py).  

The labeling is based **only on ratings (stars)** and simple patterns inside the review message.  
No natural language understanding is applied at this stage.

**Rules:**

- If the row had a numeric `rating`:
  - `≥ 4.0` → **positive**  
  - `≤ 2.0` → **negative**  
  - `2.1 – 3.9` → **neutral**

- If the row had no `rating`, but the `message` contained explicit patterns like `x/5` (e.g. *“Comida: 1/5, Serviço: 4/5, Ambiente: 5/5”*):  
  - All values are collected  
  - The **median** is calculated  
  - Converted to **positive / neutral / negative** with the same rules above

- If the row had neither `rating` nor `x/5` patterns in the `message`:  
  - `label = None`  
  - The row is discarded in the final output  

⚠️ **Important:** at this stage the **free text message is not used** (e.g., *“o atendimento foi horrível”*, *“doces maravilhosos”*).  
That’s why we later **train an NLP classifier**: to make the model understand the sentiment from text alone, even without ratings.

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

## 📌 End-to-end Sentiment Analysis Pipeline

This project builds a sentiment classifier (positive / neutral / negative) for Google Reviews.
The workflow has 4 main steps:

1️⃣ Generate Labels (Heuristic)

Create reviews_labeled.csv from the raw dataset using the labeling script:

# Linux / macOS
```bash
python -m src.data.heuristics_labeling \
  --in "data/raw/data-google-reviews.csv" \
  --out "data/processed/reviews_labeled.csv"
```

# Windows PowerShell
``` powershell
python -m src.data.heuristics_labeling --in "data/raw/data-google-reviews.csv" --out "data/processed/reviews_labeled.csv"
```

➡️ Output: data/processed/reviews_labeled.csv with column label.

2️⃣ Train the TensorFlow Model

```bash
python -m src.models.classifier_tf
```
✔️ Output:
Model → artifacts/models/keras/model_full/
Metrics → artifacts/models/keras/train_metrics.json

3️⃣ Inference (Use the Trained Model)

``` python
from src.models.classifier_tf import SentimentClassifierTF

clf = SentimentClassifierTF("artifacts/models/keras/model_full")
clf.load()

preds = clf.predict([
    "Excelente atendimento e doces incríveis!",
    "Preço razoável, nada demais.",
    "Demorou muito e veio errado."
])
print(preds)  # ['positive', 'neutral', 'negative']

```

4️⃣ Evaluation (Accuracy, F1, Confusion Matrix)

``` python
from pathlib import Path
import pandas as pd
from src.models.classifier_tf import SentimentClassifierTF
from src.eval.metrics_classification import compute_metrics, save_metrics

df = pd.read_csv("data/processed/reviews_labeled.csv")
texts, y_true = df["message"], df["label"]

clf = SentimentClassifierTF("artifacts/models/keras/model_full")
clf.load()
y_pred = clf.predict(texts)

m = compute_metrics(y_true, y_pred)
save_metrics(m, Path("artifacts/predictions"), "keras_fullset")

print("accuracy:", m["accuracy"])

```

✔️ Output:
artifacts/predictions/keras_fullset_metrics.json
artifacts/predictions/keras_fullset_confusion_matrix.csv

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

**from the repository root**
``` bash
python -m venv .venv
Activate the venv
```

🪟 Windows
``` powershell
.venv\Scripts\activate
```

🍎 macOS / 🐧 Linux
``` bash
source .venv/bin/activate

Install deps
pip install -r requirements.txt
```

**▶️ Run the Tests**
#run the whole suite
``` bash
python -m pytest -q
```

**🛠️ Troubleshooting**

🐍 ModuleNotFoundError: No module named 'src'
Make sure you run from the repo root and use the module form:
``` bash
python -m pytest -q
```

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
