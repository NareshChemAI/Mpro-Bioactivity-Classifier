import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Mpro Bioactivity Predictor",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 SARS-CoV-2 Mpro Bioactivity Predictor")
st.markdown("Predict whether a molecule is *Active* or *Inactive* against SARS-CoV-2 Main Protease")
st.markdown("---")

@st.cache_resource
def train_model():
    df = pd.read_csv("Mpro_bioactivity_v2.csv")
    def smiles_to_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        return list(fp)
    df["fp"] = df["smiles"].apply(smiles_to_fp)
    df = df.dropna(subset=["fp"])
    X = np.array(df["fp"].tolist())
    y = np.array(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train_s)
    return model

st.info("Loading model... please wait!")
model = train_model()
st.success("Model ready! ✅")

st.subheader("Enter Molecule SMILES")
smiles = st.text_input(
    "SMILES String",
    placeholder="CC(C)CC(NC(=O)c1cncc(C#N)c1)C(=O)NC1CC(=O)NC1=O"
)

if st.button("🔬 Predict Bioactivity"):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES!")
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fp_array = np.array(fp).reshape(1, -1)
            prediction = model.predict(fp_array)[0]
            probability = model.predict_proba(fp_array)[0]
            st.subheader("Result")
            if prediction == 1:
                st.success("✅ ACTIVE — This molecule INHIBITS Mpro!")
            else:
                st.error("❌ INACTIVE — This molecule does NOT inhibit Mpro")
            st.metric("Active Probability", f"{probability[1]*100:.1f}%")
            st.progress(float(probability[1]))
    else:
        st.warning("Please enter a SMILES string!")

st.markdown("---")
st.markdown("Built by *Naresh* | MSc Organic Chemistry | AI/ML in Drug Discovery")
