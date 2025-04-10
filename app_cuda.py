import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Only if using LoRA or other PEFT approach

# 1) Caching model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model(base_model_path: str, lora_checkpoint: str = None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load model (8-bit or 4-bit quantization if needed)
    # If you used BitsAndBytesConfig, adapt accordingly:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto"  
    )

    # If you're using a PEFT adapter (LoRA), load it:
    if lora_checkpoint:
        model = PeftModel.from_pretrained(model, lora_checkpoint)

    model.eval()
    return model, tokenizer

# 2) Initialize Streamlit app
st.title("My Fine-Tuned Finance Model")

# Path to base model & LoRA adapter

base_model_path = "meta-llama/Llama-3.2-3B"  
lora_checkpoint = "llama3b-lora-checkpoint"  # or None if not using LoRA

# Load model & tokenizer
model, tokenizer = load_model(base_model_path, lora_checkpoint)

# 3) UI for user input
prompt = st.text_area("Enter your prompt", value="", height=100)

generate_button = st.button("Generate")

# 4) Inference on button click
if generate_button and prompt.strip():
    with st.spinner("Generating response..."):
        # Convert prompt to tokens, move to the same device as model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode the output tokens
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Display the response
    st.write("### Response:")
    # Optionally strip out the prompt part if the model replicates it
    st.write(output_text[len(prompt):].strip())  

