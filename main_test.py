import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Streamlit to handle Koyeb's environment
if os.getenv('PORT'):
    port = int(os.getenv('PORT'))
else:
    port = 8501

def load_model():
    try:
        model_name = "TheBloke/falcon-7b-instruct-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def main():
    st.title("Faire-Appel.fr")
    st.write("Testing model loading...")
    result = load_model()
    st.write(result)

if __name__ == "__main__":
    main()