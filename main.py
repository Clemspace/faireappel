import streamlit as st
from pathlib import Path
import pypdf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dataclasses import dataclass
from typing import List, Optional, Dict
import re
from tqdm import tqdm
import tempfile
import os

def check_dependencies():
    try:
        import transformers
        st.sidebar.success(f"Transformers version: {transformers.__version__}")
    except ImportError:
        st.error("Transformers not found. Installing required packages...")
        os.system("pip install -U transformers")
        st.experimental_rerun()

    try:
        import torch
        st.sidebar.success(f"PyTorch version: {torch.__version__}")
    except ImportError:
        st.error("PyTorch not found. Installing...")
        os.system("pip install torch")
        st.experimental_rerun()

@st.cache_resource
def load_model():
    model_name = "simmo/legal-llama-3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,  # Use 4-bit quantization
            torch_dtype=torch.float16
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

st.set_page_config(
    page_title="Faire-Appel.fr",
    page_icon="⚖️",
    layout="wide"
)

@dataclass
class SummaryConfig:
    max_length: int = 150
    min_length: int = 50
    do_sample: bool = True
    temperature: float = 0.7
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


@dataclass
class AnalysisPoint:
    text: str
    category: str
    confidence: float
    legal_refs: List[str] = None
    implications: List[str] = None
    
    def __post_init__(self):
        if self.legal_refs is None:
            self.legal_refs = []
        if self.implications is None:
            self.implications = []

class LegalAnalyzer:
    def __init__(self, model_name="Equall/Saul-7B-Instruct-v1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_key_points(self, text: str) -> List[AnalysisPoint]:
        prompt = f"""Analyze the following French legal decision and extract key points that could be grounds for appeal. Format each point with its category and confidence level:

{text}

Key points:"""
        
        response = self.generate_response(prompt)
        points = []
        
        # Parse response into structured points
        current_point = ""
        for line in response.split('\n'):
            if line.startswith('- '):
                if current_point:
                    points.append(self._parse_point(current_point))
                current_point = line[2:]
            else:
                current_point += " " + line.strip()
                
        if current_point:
            points.append(self._parse_point(current_point))
            
        return points
    
    def _parse_point(self, text: str) -> AnalysisPoint:
        # Basic categories for classification
        categories = {
            "procédure": ["procédure", "délai", "notification", "compétence"],
            "fond": ["preuve", "motif", "justification", "évaluation"],
            "droits": ["droit", "liberté", "protection", "garantie"],
            "forme": ["signature", "document", "formulaire", "validité"]
        }
        
        # Determine category based on keywords
        point_category = "autre"
        max_matches = 0
        
        for category, keywords in categories.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in text.lower())
            if matches > max_matches:
                max_matches = matches
                point_category = category
        
        # Calculate basic confidence score
        confidence = min(0.9, 0.5 + (max_matches * 0.1))
        
        return AnalysisPoint(
            text=text,
            category=point_category,
            confidence=confidence
        )

    def generate_appeal_strategy(self, selected_points: List[AnalysisPoint]) -> Dict:
        if not selected_points:
            return {}
            
        points_text = "\n".join(f"- {point.text}" for point in selected_points)
        
        prompt = f"""Based on these selected appeal points:

{points_text}

Generate a comprehensive appeal strategy in French including:
1. Main arguments
2. Legal references
3. Suggested evidence
4. Procedural recommendations

Response:"""
        
        strategy = self.generate_response(prompt)
        
        # Parse the strategy into sections
        sections = {}
        current_section = ""
        current_content = []
        
        for line in strategy.split('\n'):
            if line.strip().endswith(':'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()[:-1]
                current_content = []
            else:
                current_content.append(line.strip())
                
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        return sections

class LegalSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",  # Changed to a reliable model
        device: Optional[str] = None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Use the summarization pipeline directly
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device,
                tokenizer=model_name
            )
            self.tokenizer = self.summarizer.tokenizer
            self.max_chunk_length = self.tokenizer.model_max_length - 100
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            raise

    def _split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(self.tokenizer.encode(word))
            if current_length + word_length > self.max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _extract_key_sentences(self, text: str) -> List[str]:
        key_markers = [
            "CONSIDÉRANT QUE",
            "PAR CES MOTIFS",
            "ATTENDU QUE",
            "LA COUR",
            "EN CONSÉQUENCE",
            "DISPOSITIF",
            "SUR LE MOYEN",
            "ALORS QUE"
        ]
        
        sentences = text.split(". ")
        key_sentences = []
        
        for sentence in sentences:
            if any(marker in sentence.upper() for marker in key_markers):
                key_sentences.append(sentence.strip() + ".")
                
        return key_sentences

    def summarize(
        self, 
        text: str, 
        config: Optional[SummaryConfig] = None
    ) -> dict:
        if config is None:
            config = SummaryConfig()
            
        chunks = self._split_into_chunks(text)
        key_sentences = self._extract_key_sentences(text)
        
        progress_bar = st.progress(0)
        chunk_summaries = []
        
        for idx, chunk in enumerate(chunks):
            try:
                summary = self.summarizer(
                    chunk,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    early_stopping=config.early_stopping
                )[0]['summary_text']
                
                chunk_summaries.append(summary)
            except Exception as e:
                st.warning(f"Warning: Could not summarize chunk {idx + 1}: {str(e)}")
                # Add the original chunk if summarization fails
                chunk_summaries.append(chunk[:config.max_length])
            
            progress_bar.progress((idx + 1) / len(chunks))
            
        result = {
            "full_summary": " ".join(chunk_summaries),
            "key_points": key_sentences,
            "metadata": {
                "chunks_processed": len(chunks),
                "model_used": self.summarizer.model.name_or_path,
                "config": vars(config)
            }
        }
        
        return result

class LegalTextAnalyzer:
    def __init__(self):
        with st.spinner('Chargement du modèle de traitement...'):
            self.summarizer = LegalSummarizer()
        
    def analyze_decision(self, text: str) -> dict:
        with st.spinner('Analyse en cours...'):
            summary_result = self.summarizer.summarize(text)
            
            appeal_grounds = self._identify_appeal_grounds(text)
            legal_refs = self._extract_legal_references(text)
            
            return {
                "summary": summary_result["full_summary"],
                "key_points": summary_result["key_points"],
                "appeal_grounds": appeal_grounds,
                "legal_references": legal_refs
            }
        
    def _identify_appeal_grounds(self, text: str) -> List[dict]:
        grounds = []
        
        patterns = {
            "erreur_procédurale": [
                "violation des règles de procédure",
                "vice de procédure",
                "nullité de procédure",
                "irrégularité de procédure"
            ],
            "mauvaise_application_loi": [
                "violation de la loi",
                "fausse application",
                "erreur de droit",
                "mauvaise interprétation"
            ],
            "motivation_insuffisante": [
                "motivation insuffisante",
                "défaut de base légale",
                "contradiction de motifs",
                "défaut de réponse à conclusions"
            ],
            "violation_droits": [
                "violation des droits de la défense",
                "non-respect du contradictoire",
                "violation du principe du contradictoire"
            ]
        }
        
        for ground_type, indicators in patterns.items():
            matches = []
            for indicator in indicators:
                count = text.lower().count(indicator.lower())
                if count > 0:
                    matches.append({
                        "indicator": indicator,
                        "count": count
                    })
            
            if matches:
                max_count = max(m["count"] for m in matches)
                confidence = "high" if max_count > 1 else "medium"
                grounds.append({
                    "type": ground_type,
                    "matches": matches,
                    "confidence": confidence
                })
                    
        return grounds
    
    def _extract_legal_references(self, text: str) -> List[str]:
        patterns = [
            r"article [0-9]+(?:-[0-9]+)? du code\s+[a-zéèêë]+",
            r"loi n°\s*[0-9\-]+\s+du\s+[0-9]{1,2}\s+[a-zéèêë]+\s+[0-9]{4}",
            r"Cass\.\s*(?:civ|com|soc)\.?\s*(?:1re|2e|3e)?,\s*[0-9]{1,2}\s+[a-zéèêë]+\s+[0-9]{4}",
        ]
        
        references = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            references.extend(match.group(0) for match in matches)
            
        return list(set(references))

def extract_text_from_pdf(pdf_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file.flush()
        
        reader = pypdf.PdfReader(tmp_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
    os.unlink(tmp_file.name)
    return text

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
{chr(10).join(f'- {point.text}' for point in selected_points)}

## Stratégie détaillée
{chr(10).join(f'### {section}{chr(10)}{content}' for section, content in strategy.items())}
"""
if __name__ == "__main__":
    main()