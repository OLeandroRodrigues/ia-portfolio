📝 Google Reviews NLP Pipeline

This repository contains an experimental NLP pipeline to analyze and classify Google Reviews.
The project is organized into two layers:

1. Comment Classification

   🎭 Sentiment analysis (positive | neutral | negative)
   🏷️ Topic classification (e.g., service, price, quality)
   🚨 Spam/legitimacy detection (suspect | legitimate)

2. Author Linking Experiments

    🔍 Detects possible same-author patterns across different accounts
    ✍️ Based on writing fingerprint + semantic embeddings + clustering

🎯 Objectives

✅ Train and evaluate classification models using TensorFlow.
✅ Extract stylometry features and multilingual embeddings.
✅ Build similarity graphs to detect repeated authorship.
✅ Provide explainable outputs for research and audit.

📂 Project Structure
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


⚙️ Tech Stack

🐍 Python 3.10+
🔶 TensorFlow / Keras – model training
🌍 TensorFlow Hub – multilingual embeddings (USE / LaBSE)
📊 scikit-learn – clustering (DBSCAN, HDBSCAN) & metrics
📚 spaCy (pt) – optional preprocessing for Portuguese
📈 pandas / matplotlib / seaborn – analysis & visualization

🔬 Workflow

1. Exploratory Data Analysis (01_eda.ipynb)

    -   Inspect raw reviews, distributions & anomalies
    -   Apply basic cleaning

2.  Classification (02_train_classifier.ipynb)

    -   Train sentiment / topic / spam classifiers
    -   Evaluate with accuracy, F1-score & confusion matrix

3.  Author Linking (03_linking_experiments.ipynb)

    -   Generate embeddings + stylometry features
    -   Build similarity graph & cluster comments
    -   Identify candidate same-author groups

📊 Outputs

📑 Classification reports (JSON, PNG plots)
🤖 Saved TensorFlow models (.h5, .pkl)
🔗 Similarity graphs of comments
👥 Author cluster assignments (author_clusters.csv)

⚠️ Disclaimer

This project is intended for research and educational purposes only.
Author linking is probabilistic and should be interpreted as candidate same-author groups, not definitive proof.
All datasets used are either publicly available or synthetic.

🚀 Next Steps

Improve weak labeling for "suspect vs legitimate" reviews.
Train a Siamese/contrastive model for authorship verification.
Add temporal & behavioral features (time, frequency).
Create an interactive dashboard for results.