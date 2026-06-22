#🧬 SARS-CoV-2 Mpro Bioactivity Classifier

##🎯 Project Overview
Machine learning model to predict bioactivity 
of molecules against SARS-CoV-2 Main Protease 
(Mpro) — a key drug target for COVID-19.

##🧪 Problem Statement
- Target: SARS-CoV-2 Main Protease (Mpro)
- Task: Binary classification
- Active (1): pIC50 >= 6
- Inactive (0): pIC50 < 6

##📊 Dataset
- Source: ChEMBL Database (CHEMBL4303835)
- Total molecules: 4946
- Active: 195 | Inactive: 4751

## ⚙ Workflow
1.📥 Data collection from ChEMBL
2.🧹 Data cleaning and preprocessing
3.🔢 IC50 to pIC50 conversion
4.🔬 Morgan Fingerprint generation (2048 bits)
5.✂ Train-Test Split (80/20, stratified)
6.⚖ SMOTE for class imbalance handling
7.📳 Model training (Random Forest + XGBoost)
8.📉 Model evaluation

##🏆 Results
| Model | AUC-ROC | F1 Score | MCC |
|-------|---------|----------|-----|
|🌲 Random Forest | 0.999 | 0.940 | 0.939 |
|⚡ XGBoost | 1.000 | - | - |

##🛠 Libraries Used
- Python, RDKit, Scikit-learn
- XGBoost, Imbalanced-learn
- Pandas, NumPy, Joblib

## Author
*Naresh — MSc Organic Chemistry*
AI/ML in Drug Discovery
