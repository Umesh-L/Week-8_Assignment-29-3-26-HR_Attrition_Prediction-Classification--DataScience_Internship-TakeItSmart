# HR Attrition Prediction (Classification)

Predict employee attrition (Yes/No) from HR features such as age, compensation, role, satisfaction, and tenure. The full workflow lives in the notebook and includes EDA, preprocessing, modeling, and model selection. Built for the Week-8 Assignment (29-3-26) of the Take It Smart Data Science internship.

## Repository contents
- Notebook: [HR_Attrition_Prediction_(Classificaiton).ipynb](HR_Attrition_Prediction_(Classificaiton).ipynb)
- Data: [Dataset and Task pdf/HR-Employee-Attrition.csv](Dataset%20and%20Task%20pdf/HR-Employee-Attrition.csv)
- Supporting assets: Output snapshots/ (plots and metrics), Screen recording/ (walkthrough video)

## Environment setup
- Python 3.9+ recommended.
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
  ```

## How to run
1. Keep the dataset at Dataset and Task pdf/HR-Employee-Attrition.csv relative to the notebook.
2. Start Jupyter (or open directly in VS Code):
  ```bash
  jupyter notebook
  ```
3. Open the notebook and run cells top-to-bottom to reproduce the analysis and metrics.

## Workflow (as implemented in the notebook)
- Data loading and inspection: shape, dtypes, summary stats; confirms no missing values; drops duplicates if found.
- Exploratory analysis: univariate (Age, BusinessTravel), bivariate (Attrition vs MonthlyIncome, Department), correlation heatmap for numeric features.
- Outlier handling: caps MonthlyIncome using IQR upper bound.
- Encoding and scaling: label-encodes Attrition; one-hot encodes remaining categoricals (drop_first); standard-scales all feature columns.
- Split and balance: stratified 80/20 train-test split; applies SMOTE on training data to balance classes.
- Models trained: Logistic Regression, Decision Tree, Random Forest, KNN.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC on the test set; confusion matrices printed per model.
- Cross-validation: 5-fold CV (F1 scoring) on the SMOTE-balanced training set for each model.
- Hyperparameter tuning: GridSearchCV on Random Forest (n_estimators, max_depth, min_samples_split).
- Model selection: Logistic Regression chosen for best generalization to test data.

## Assets
- Output screenshots:
  - <img width="689" height="495" alt="age distribution graph" src="https://github.com/user-attachments/assets/a7a89e77-4081-4896-b1ab-1f5e182d9a08" />
  - <img width="700" height="507" alt="business travel count graph" src="https://github.com/user-attachments/assets/b327b586-da36-4b7f-872d-b6725b6a5a92" />
  - <img width="712" height="499" alt="monthly income vs attrition graph" src="https://github.com/user-attachments/assets/fed9ce5a-e8e7-4234-9326-f114d17ba192" />
  - <img width="886" height="512" alt="department vs attrition graph" src="https://github.com/user-attachments/assets/e93ad8a9-ab76-4b83-91a3-b3616a777b7b" />
  - <img width="903" height="713" alt="correlation heatmap" src="https://github.com/user-attachments/assets/075e792e-13e2-46ad-8a06-50e44fb3e480" />
  - <img width="505" height="351" alt="evaluate_models-confusion matrix" src="https://github.com/user-attachments/assets/3d2e446e-8438-4fac-bb01-bd53f9bc4235" />
  - <img width="409" height="110" alt="cross validation result" src="https://github.com/user-attachments/assets/b47d4e93-d3b2-44eb-953d-b3a78f589841" />
  - <img width="849" height="234" alt="model performance comparision table" src="https://github.com/user-attachments/assets/ba76be78-505a-4305-879e-5d93c971ec5d" />
  - <img width="885" height="255" alt="best model" src="https://github.com/user-attachments/assets/dd1ac807-ff4f-4886-b41e-658f1a161b0b" />

- Explanation video:
https://github.com/user-attachments/assets/0e65447b-4046-4525-9f18-811786d87525

## Key results
- Logistic Regression (final choice): Accuracy ≈ 0.82, F1 ≈ 0.46, Recall ≈ 0.62; best balance and recall on test data.
- Random Forest (initial and tuned): strong CV F1 (~0.93 initial) but lower test F1 (~0.35–0.39), indicating overfitting.
- SMOTE balancing and F1-focused CV were used to address class imbalance during training.

## EDA highlights
- Compensation is a major driver: lower MonthlyIncome strongly links to attrition; MonthlyIncome, JobLevel, and TotalWorkingYears are tightly correlated.
- Departmental risk: Sales shows a higher attrition rate relative to its size; R&D has more departures by volume because it is larger.
- Travel and workload: most employees travel rarely; overtime and weaker work-life balance flag churn risk.
- Tenure patterns: YearsAtCompany, YearsInCurrentRole, and YearsWithCurrManager are highly correlated, suggesting limited internal movement.
