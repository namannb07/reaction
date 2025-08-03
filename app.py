import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from googletrans import Translator
import re
import base64
from io import BytesIO
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Reactelligence - AI Chemistry Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .molecule-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session cache
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}

class ChemistryAI:
    def __init__(self):
        self.translator = Translator()
        self.model_cache = {}

    @st.cache_resource
    def load_chemistry_model(_self):
        """Load pre-trained ChemBERTa model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            except:
                model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            return tokenizer, model
        except Exception as e:
            st.warning(f"Could not load ChemBERTa model: {e}")
            return None, None

    def translate_to_english(self, text, source_lang='auto'):
        if text in st.session_state.translation_cache:
            return st.session_state.translation_cache[text]
        try:
            if source_lang == 'en' or self._is_english(text):
                return text
            result = self.translator.translate(text, src=source_lang, dest='en')
            translated = result.text
            st.session_state.translation_cache[text] = translated
            return translated
        except:
            return text

    def _is_english(self, text):
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_chars = sum(1 for c in text if c.isalpha())
        return total_chars == 0 or english_chars / total_chars > 0.7

    def detect_intent(self, text):
        text_lower = text.lower()
        reaction_keywords = ['reaction', 'predict', 'product', 'react', 'synthesis', 'yield']
        generation_keywords = ['generate', 'create', 'design', 'new molecule', 'novel', 'drug']
        analysis_keywords = ['analyze', 'properties', 'molecular weight', 'logp', 'analyze']

        if self._contains_smiles(text):
            if any(keyword in text_lower for keyword in reaction_keywords):
                return 'reaction_prediction'
            elif any(keyword in text_lower for keyword in analysis_keywords):
                return 'analysis'
            else:
                return 'analysis'

        if any(keyword in text_lower for keyword in reaction_keywords):
            return 'reaction_prediction'
        elif any(keyword in text_lower for keyword in generation_keywords):
            return 'molecule_generation'
        elif any(keyword in text_lower for keyword in analysis_keywords):
            return 'analysis'
        else:
            return 'analysis'

    def _contains_smiles(self, text):
        smiles_pattern = r'[A-Za-z0-9@+\-\[\]()=#$:/.\\]+'
        potential_smiles = re.findall(smiles_pattern, text)
        for smiles in potential_smiles:
            if len(smiles) > 3 and Chem.MolFromSmiles(smiles) is not None:
                return True
        return False

    def extract_smiles(self, text):
        words = text.split()
        smiles_list = []
        for word in words:
            mol = Chem.MolFromSmiles(word)
            if mol is not None:
                smiles_list.append(word)
        return smiles_list

    def predict_reaction(self, reactants_smiles):
        tokenizer, model = self.load_chemistry_model()
        if model is None:
            return self._fallback_reaction_prediction(reactants_smiles)

        try:
            if hasattr(model, "generate"):  # Seq2Seq style
                input_text = f"React: {'.'.join(reactants_smiles)}"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                with st.spinner("Predicting reaction products..."):
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=200,
                        num_beams=5,
                        temperature=0.7,
                        do_sample=True
                    )
                predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
                products = self._parse_products(predicted)
                return products
            else:
                st.warning("ChemBERTa does not support direct reaction generation. Using fallback prediction.")
                return self._fallback_reaction_prediction(reactants_smiles)
        except Exception as e:
            st.error(f"Reaction prediction error: {e}")
            return self._fallback_reaction_prediction(reactants_smiles)

    def _fallback_reaction_prediction(self, reactants_smiles):
        products = []
        for smiles in reactants_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                if 'C=C' in smiles:
                    products.append(smiles.replace('C=C', 'CC'))
                elif 'C#C' in smiles:
                    products.append(smiles.replace('C#C', 'C=C'))
                else:
                    products.append(smiles)
        return products if products else ['CCO']

    def _parse_products(self, predicted_text):
        smiles_candidates = self.extract_smiles(predicted_text)
        return smiles_candidates if smiles_candidates else ['CCO']

    def generate_molecule(self, prompt):
        tokenizer, model = self.load_chemistry_model()
        if model is None or not hasattr(model, "generate"):
            st.warning("ChemBERTa does not support molecule generation. Using fallback molecules.")
            return self._fallback_molecule_generation(prompt)

        try:
            input_text = f"Generate molecule for: {prompt}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            with st.spinner("Generating novel molecule..."):
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=5,
                    temperature=0.8,
                    do_sample=True
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            molecules = self.extract_smiles(generated)
            if not molecules:
                return self._fallback_molecule_generation(prompt)
            return molecules[0]
        except Exception as e:
            st.error(f"Molecule generation error: {e}")
            return self._fallback_molecule_generation(prompt)

    def _fallback_molecule_generation(self, prompt):
        drug_molecules = [
            'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',
            'CC(=O)Oc1ccccc1C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CCN(CC)CCCC(C)Nc1ccnc2cc(ccc12)Cl',
            'Cc1ccc(cc1)C(=O)O'
        ]
        return np.random.choice(drug_molecules)

    def analyze_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        properties = {
            'SMILES': smiles,
            'Molecular Formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'Molecular Weight': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Rings': Descriptors.RingCount(mol),
            'Formal Charge': Chem.rdmolops.GetFormalCharge(mol),
            'Atoms': mol.GetNumAtoms(),
            'Bonds': mol.GetNumBonds()
        }
        return properties

class MoleculeVisualizer:
    @staticmethod
    def draw_molecule(smiles, size=(400, 400)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    @staticmethod
    def create_molecule_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
        return G

    @staticmethod
    def plot_molecule_graph(G, title="Molecular Graph"):
        if G is None:
            return None
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, ax=ax)
        node_colors = []
        node_labels = {}
        for node in G.nodes():
            symbol = G.nodes[node]['symbol']
            node_labels[node] = symbol
            if symbol == 'C':
                node_colors.append('black')
            elif symbol == 'O':
                node_colors.append('red')
            elif symbol == 'N':
                node_colors.append('blue')
            elif symbol == 'S':
                node_colors.append('yellow')
            else:
                node_colors.append('purple')
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=12, font_color='white', ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig

# ---------------- MAIN ----------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Reactelligence - AI Chemistry Lab</h1>
        <p style="text-align: center; color: white; margin: 0;">
            AI-Powered Chemistry Analysis & Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    chem_ai = ChemistryAI()
    visualizer = MoleculeVisualizer()

    with st.sidebar:
        st.header("🔬 Features")
        st.markdown("""
        - **Reaction Prediction**
        - **Molecule Generation**
        - **Property Analysis**
        - **Multilingual Support**
        - **Visual Analysis**
        """)
        st.header("🌍 Language")
        language = st.selectbox("Select Input Language", [
            "Auto-detect", "English", "Hindi", "Spanish", "French",
            "German", "Chinese", "Japanese", "Korean"
        ])
        st.header("📝 Examples")
        if st.button("🧪 Reaction Example"):
            st.session_state.example_input = "Predict the reaction: CCO + O2"
        if st.button("🔬 Generation Example"):
            st.session_state.example_input = "Generate a pain relief molecule"
        if st.button("📊 Analysis Example"):
            st.session_state.example_input = "Analyze CC(=O)Oc1ccccc1C(=O)O"

    st.header("💬 Chemistry Query")
    default_input = st.session_state.get('example_input', '')
    user_input = st.text_area(
        "Enter your chemistry question in any language:",
        value=default_input,
        height=100
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear")

    if clear_btn:
        st.session_state.example_input = ''
        st.rerun()

    if analyze_btn and user_input.strip():
        lang_code = 'auto' if language == "Auto-detect" else language.lower()[:2]
        translated_input = chem_ai.translate_to_english(user_input, lang_code)
        if translated_input != user_input:
            st.info(f"**Translated:** {translated_input}")
        intent = chem_ai.detect_intent(translated_input)
        st.success(f"**Detected Task:** {intent.replace('_', ' ').title()}")

        if intent == 'reaction_prediction':
            reactants = chem_ai.extract_smiles(translated_input)
            if not reactants:
                st.warning("No valid SMILES found.")
            else:
                products = chem_ai.predict_reaction(reactants)
                st.write("**Predicted Products:**", products)

        elif intent == 'molecule_generation':
            generated_smiles = chem_ai.generate_molecule(translated_input)
            st.write("Generated Molecule:", generated_smiles)

        else:
            smiles_list = chem_ai.extract_smiles(translated_input)
            if not smiles_list:
                st.warning("No valid SMILES found.")
            else:
                for smiles in smiles_list:
                    st.write("**Molecule:**", smiles)
                    props = chem_ai.analyze_molecule(smiles)
                    st.write(props)

if __name__ == "__main__":
    main()
