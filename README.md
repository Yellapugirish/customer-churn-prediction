# Customer Churn Prediction (Telco)

End-to-end EDA and baseline machine learning model to predict churn on the Telco Customer Churn dataset (Jupyter/Anaconda).

## What’s inside
- Data cleaning (TotalCharges → numeric, handle nulls)
- Categorical encoding
- Train/test split
- Baseline **RandomForest** classifier
- Evaluation: **accuracy, F1**, confusion matrix, (optional) ROC-AUC
- Feature importances

## Files
- `customer-churn.ipynb` — main notebook (run top→bottom)
- `customer_churn.csv` — dataset
- `requirements.txt` — dependencies
- `.gitignore` — ignore checkpoints/caches

## Run (Anaconda/Jupyter)
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `customer-churn.ipynb` and **Run All**.

## Current Results (fill from your notebook)
- Accuracy: **___%**
- ROC-AUC: **___**
(See notebook for full classification report & confusion matrix.)

## Notes / Next steps
- Try one-hot encoding vs label encoding and compare
- Use `class_weight='balanced'` or SMOTE for class imbalance
- Simple tuning (`n_estimators`, `max_depth`) or try Logistic Regression/XGBoost
- Add cross-validation for robust metrics
