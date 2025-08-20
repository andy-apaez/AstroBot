import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import streamlit as st

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    st.error("Please set HF_TOKEN in your .env file!")
    st.stop()

# Login to Hugging Face
login(token=hf_token)

# ----------------------------
# Load LLaMA 3 Model with disk offload (safe for CPU)
# ----------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},       
    offload_folder="offload",     
    offload_state_dict=True,      
    torch_dtype=torch.float32,
    use_auth_token=True
)

# ----------------------------
# Create pipeline
# ----------------------------
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9
)

# ----------------------------
# System prompt
# ----------------------------
system_prompt = """You are AstroBot 🚀, an AI expert in astronomy, astrophysics, and space exploration.
You ONLY answer questions about space facts, theories, and discoveries.
If asked something unrelated to space, politely reply:
'I only talk about space topics 🌌'."""

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AstroBot", page_icon="🚀")
st.title("AstroBot 🌌")
st.write("Ask me anything about space!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# User input
user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    prompt = f"{system_prompt}\nUser: {user_input}\nAstroBot:"
    response = chatbot(prompt)[0]['generated_text'].split("AstroBot:")[-1].strip()
    
    # Save chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AstroBot", response))
    st.experimental_rerun()

# Display chat history with styled chat bubbles
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"""
        <div style="background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px; max-width:80%; float:right; clear:both;">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#EAEAEA; padding:10px; border-radius:10px; margin:5px; max-width:80%; float:left; clear:both;">
            <strong>AstroBot:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

# Add spacing at the bottom so last message isn't hidden
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
