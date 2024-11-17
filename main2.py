import streamlit as st
from pathlib import Path
import pypdf
from typing import List, Optional, Dict
from dataclasses import dataclass
import tempfile
import os
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    Document,
    SimpleDirectoryReader,
    PromptHelper,
)
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

@dataclass
class AnalysisPoint:
    text: str
    category: str
    confidence: float
    legal_refs: List[str] = None
    
    def __post_init__(self):
        if self.legal_refs is None:
            self.legal_refs = []

class LocalLLM:
    def __init__(self):
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model and tokenizer
        model_name = "TheBloke/falcon-7b-instruct-GPTQ"  # We can also use "TheBloke/mpt-7b-instruct-4bit-128g"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

class LegalAnalyzer:
    def __init__(self):
        # Initialize local LLM
        local_llm = LocalLLM()
        
        # Create HuggingFaceLLM instance for LlamaIndex
        self.llm = HuggingFaceLLM(
            tokenizer=local_llm.tokenizer,
            model=local_llm.model,
            max_new_tokens=512,
            temperature=0.1,
            device_map="auto",
            model_kwargs={"torch_dtype": torch.float16}
        )
        
        # Use all-mpnet-base for embeddings (good balance of performance and quality)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Configure prompt helper
        self.prompt_helper = PromptHelper(
            max_input_size=2048,
            num_output=512,
            max_chunk_overlap=20
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            prompt_helper=self.prompt_helper
        )

    def process_document(self, text: str) -> VectorStoreIndex:
        documents = [Document(text=text)]
        return VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )

    def extract_key_points(self, index: VectorStoreIndex) -> List[AnalysisPoint]:
        query_engine = index.as_query_engine()
        
        prompt = """Analyse cette décision de justice française et identifie les points clés 
        qui pourraient justifier un appel. Pour chaque point identifié, indique:
        1. Le point spécifique qui pose problème
        2. La catégorie (procédure, fond, droits, forme)
        3. Les articles de loi ou jurisprudences pertinents

        Format de réponse souhaité:
        Point: [description du problème]
        Catégorie: [type]
        Articles: [références]
        """
        
        response = query_engine.query(prompt)
        points = []
        current_point = {}
        
        for line in str(response).split('\n'):
            line = line.strip()
            if line.startswith('Point:'):
                if current_point:
                    points.append(self._create_analysis_point(current_point))
                current_point = {'text': line.split(':', 1)[1].strip()}
            elif line.startswith('Catégorie:'):
                current_point['category'] = line.split(':', 1)[1].strip()
            elif line.startswith('Articles:'):
                current_point['refs'] = line.split(':', 1)[1].strip()
        
        if current_point:
            points.append(self._create_analysis_point(current_point))
            
        return points

    def _create_analysis_point(self, point_dict: dict) -> AnalysisPoint:
        return AnalysisPoint(
            text=point_dict.get('text', ''),
            category=point_dict.get('category', 'autre'),
            confidence=0.8 if point_dict.get('refs') else 0.6,
            legal_refs=point_dict.get('refs', '').split(', ') if point_dict.get('refs') else []
        )

    def generate_appeal_strategy(self, index: VectorStoreIndex, selected_points: List[AnalysisPoint]) -> Dict:
        if not selected_points:
            return {}
        
        points_text = "\n".join(f"- {point.text}" for point in selected_points)
        query_engine = index.as_query_engine()
        
        prompt = f"""En te basant sur ces moyens d'appel:
        {points_text}
        
        Génère une stratégie d'appel détaillée en français avec:
        1. Arguments juridiques principaux
        2. Références légales et jurisprudentielles
        3. Éléments de preuve à rassembler
        4. Recommandations procédurales
        """
        
        response = query_engine.query(prompt)
        
        sections = {
            "Arguments juridiques": [],
            "Références juridiques": [],
            "Éléments de preuve": [],
            "Recommandations": []
        }
        
        current_section = None
        for line in str(response).split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if any(line.startswith(str(i)) for i in range(1, 5)):
                section_map = {
                    "1": "Arguments juridiques",
                    "2": "Références juridiques",
                    "3": "Éléments de preuve",
                    "4": "Recommandations"
                }
                current_section = section_map.get(line[0])
            elif current_section and line.startswith('-'):
                sections[current_section].append(line[1:].strip())
            elif current_section:
                sections[current_section].append(line)
                
        return sections

def main():
    st.title("Faire-Appel.fr ⚖️")
    st.subheader("Assistant d'analyse des décisions de justice")
    
    analyzer = LegalAnalyzer()
    
    uploaded_file = st.file_uploader("Déposez votre décision de justice (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner('Analyse du document en cours...'):
            # Extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()
                
                reader = pypdf.PdfReader(tmp_file.name)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
            os.unlink(tmp_file.name)
            
            # Extract key points
            points = analyzer.extract_key_points(text)
            
            # Display points for selection
            st.header("Points clés identifiés")
            
            selected_points = []
            for point in points:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.checkbox(
                        point.text, 
                        help=f"Catégorie: {point.category}, Confiance: {point.confidence:.2%}"
                    ):
                        selected_points.append(point)
                with col2:
                    st.info(f"{point.category}")
            
            if selected_points:
                st.header("Stratégie d'appel")
                if st.button("Générer la stratégie d'appel"):
                    with st.spinner('Génération de la stratégie...'):
                        strategy = analyzer.generate_appeal_strategy(selected_points)
                        
                        # Display strategy sections
                        for section, content in strategy.items():
                            with st.expander(section):
                                st.markdown(content)
                        
                        # Option to download
                        if st.button("Télécharger le rapport"):
                            report = f"""# Rapport de stratégie d'appel
                            
## Points sélectionnés
{''.join(f'- {point.text}\n' for point in selected_points)}

## Stratégie détaillée
{''.join(f'### {section}\n{content}\n\n' for section, content in strategy.items())}
"""
                            st.download_button(
                                "Télécharger en format texte",
                                report,
                                file_name="strategie_appel.txt"
                            )

if __name__ == "__main__":
    main()