# Heart Disease Prediction — ML Workflow

## Structure
- `data/raw/heart.csv` : Place original dataset here.
- `data/processed/heart_cleaned.csv` : Cleaned dataset (saved after preprocessing).
- `notebooks/01_eda.ipynb` : Exploratory Data Analysis.
- `notebooks/02_modeling.ipynb` : Modeling experiments.
- `src/` : Source scripts (`train.py`, `predict.py`, preprocessing helpers).
- `results/figures` : Evaluation plots.
- `results/models` : Saved model files.

## Quick start
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
