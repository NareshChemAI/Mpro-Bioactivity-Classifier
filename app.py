import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Mpro Bioactivity Predictor",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 SARS-CoV-2 Mpro Bioactivity Predictor")
st.markdown("Predict whether a molecule is *Active* or *Inactive* against SARS-CoV-2 Main Protease")
st.markdown("---")

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

model = load_model()

st.subheader("Enter Molecule SMILES")
smiles = st.text_input(
    "SMILES String",
    placeholder="Example: CC(C)CC(NC(=O)c1cncc(C#N)c1)C(=O)NC1CC(=O)NC1=O"
)

if st.button("🔬 Predict Bioactivity"):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES! Please enter a valid molecule.")
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fp_array = np.array(fp).reshape(1, -1)
            prediction = model.predict(fp_array)[0]
            probability = model.predict_proba(fp_array)[0]
            img = Draw.MolToImage(mol, size=(300, 300))
            st.subheader("Molecule Structure")
            st.image(img, caption="Input Molecule")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success("✅ ACTIVE — This molecule INHIBITS Mpro!")
                st.metric("Active Probability", f"{probability[1]*100:.1f}%")
            else:
                st.error("❌ INACTIVE — This molecule does NOT inhibit Mpro")
                st.metric("Active Probability", f"{probability[1]*100:.1f}%")
            st.subheader("Confidence Score")
            st.progress(float(probability[1]))
    else:
        st.warning("Please enter a SMILES string!")

st.markdown("---")
st.markdown("Built by *Naresh* | MSc Organic Chemistry | AI/ML in Drug Discovery")
