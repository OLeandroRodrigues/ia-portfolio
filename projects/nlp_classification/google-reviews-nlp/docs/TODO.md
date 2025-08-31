# ✅ TODO List — Google Reviews NLP Pipeline

This file tracks **pending tasks, improvements, and ideas** for the project.  
It is a lightweight backlog — not a substitute for an issue tracker, but useful for quick notes.

---

## 📌 Core Tasks
- [ ] Improve **heuristic labeling** rules  
  - Handle sarcasm / mismatched rating vs. message  
  - Detect missing `rating` with only textual clues
- [ ] Finalize **classifier training** script (`classifier_tf.py`)  
  - Add CLI arguments (`--epochs`, `--batch-size`, `--class-weight`)  
  - Enable saving checkpoints
- [ ] Add **Hugging Face baseline** comparison (DistilBERT, mBERT)  
- [ ] Integrate **evaluation metrics** with plots (confusion matrix heatmap)

---

## 🧪 Testing
- [ ] Add unit tests for `heuristics_labeling.py` (parse ratings properly)  
- [ ] Add smoke test for `SentimentClassifierTF` (train small dataset → predict)  
- [ ] Test `SentimentClassifierHF` with different models

---

## 📊 Analysis & Notebooks
- [ ] Expand `02_train_classifier.ipynb` with:  
  - Training curves visualization (accuracy/loss)  
  - Per-class F1-scores  
- [ ] Add notebook to compare **heuristic labels vs. model predictions**  
- [ ] Document class imbalance strategies

---

## 🔗 Author Linking (Next Phase)
- [ ] Implement `build_similarity_graph.py` end-to-end  
- [ ] Add DBSCAN/HDBSCAN experiments  
- [ ] Evaluate stylometry + embeddings combination

---

## ⚙️ DevOps & Repo Hygiene
- [ ] Add **Makefile / Taskfile** with shortcuts (`make label-data`, `make train`, etc.)  
- [ ] Create **Dockerfile** for reproducible environment  
- [ ] Add CI (GitHub Actions) to run tests automatically on push  
- [ ] Split requirements into `requirements.txt` (core) and `requirements-dev.txt` (extras: black, ruff, transformers)

---

## 💡 Ideas (Future Work)
- [ ] Dashboard (Streamlit or Gradio) for interactive sentiment demo  
- [ ] Weak supervision to combine ratings + Hugging Face predictions  
- [ ] Contrastive/Siamese model for authorship verification  
- [ ] Multilingual extension (reviews in PT/EN/ES)  

---

👉 Keep this file updated as tasks progress.  
Completed items can be checked `[x]` or moved to a `DONE` section at the bottom.