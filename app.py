import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset
import os

# -----------------------------------------
# BACKEND FUNCTIONS (Requirement #2 & #3)
# -----------------------------------------

BASE_MODEL = "gpt2"
OUTPUT_DIR = "./fine_tuned_gpt2"

@st.cache_resource
def load_base_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def preprocess_and_train(dataset_path: str, epochs: int, batch_size: int):
    """Handles text preprocessing, tokenization, and the training loop."""
    tokenizer = load_base_tokenizer()
    
    # 1. Preprocessing
    dataset = load_dataset('text', data_files={'train': dataset_path})
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 2. Model Loading
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    
    # 3. Training Loop
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=5,
        save_strategy="no",
        report_to="none" # Prevents wandb logging errors
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return trainer.state.global_step # Return a metric for the frontend

def generate_text(prompt: str):
    """Inference generation for both base and fine-tuned models."""
    tokenizer = load_base_tokenizer()
    device = 0 if torch.cuda.is_available() else -1
    
    # Base Inference
    base_gen = pipeline("text-generation", model=BASE_MODEL, tokenizer=tokenizer, device=device)
    base_out = base_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    
    # Fine-Tuned Inference
    try:
        tuned_gen = pipeline("text-generation", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR, device=device)
        tuned_out = tuned_gen(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    except Exception:
        tuned_out = "Error: Fine-tuned model not found. Please train first."
        
    return base_out, tuned_out

# -----------------------------------------
# FRONTEND INTERFACE (Requirement #1 & #4)
# -----------------------------------------

st.set_page_config(page_title="GPT-2 Fine-Tuner", layout="wide")
st.title("GPT-2 Custom Fine-Tuner")
st.markdown("Fine-tune GPT-2 on your custom text dataset using PyTorch and Hugging Face.")

# Requirement #1: Meaningful Controls (File uploader, Sliders, Text Input)
st.sidebar.header("1. Training Settings")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (.txt)", type=["txt"])
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=10, value=3)
batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=8, value=2)
start_training = st.sidebar.button("Start Fine-Tuning")

if start_training:
    if uploaded_file is not None:
        # Save file locally for the backend
        dataset_path = "dataset.txt"
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Training in progress... Check your terminal for loss metrics."):
            try:
                # Call backend function
                total_steps = preprocess_and_train(dataset_path, epochs, batch_size)
                st.sidebar.success(f"Training Complete! Total steps processed: {total_steps}")
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
    else:
        st.sidebar.warning("Please upload a dataset first.")

st.header("2. Test & Compare Models")
prompt = st.text_input("Enter a starting prompt:", "The LSTM model forecasted")

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        base_result, tuned_result = generate_text(prompt)
        
        # Requirement #4: Display output and meaningful comparison/diagnostic
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base GPT-2 Output")
            st.info(base_result)
            
        with col2:
            st.subheader("Fine-Tuned Output")
            st.success(tuned_result)
            
        st.caption("Diagnostic Comparison: Notice how the fine-tuned model adopts the specific vocabulary and context of your uploaded dataset compared to the base model's generalized output.")