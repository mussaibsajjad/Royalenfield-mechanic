# app.py
import streamlit as st
from dotenv import load_dotenv
from rag import rag_answer  # our new helper

# load .env so streamlit picks up your AZURE_* vars
load_dotenv()

st.set_page_config(
    page_title="Royal Enfield Mechanic Assistant",
    layout="wide",
)

st.title("🔧 Royal Enfield Mechanic Assistant")
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of context chunks (k)", min_value=1, max_value=8, value=4)

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Ask a question about your Royal Enfield…")
if st.button("🔍 Ask"):
    if prompt:
        with st.spinner("Thinking…"):
            answer = rag_answer(prompt, k=k)
        st.session_state.history.append((prompt, answer))

st.markdown("## Conversation")
for q, a in st.session_state.history[::-1]:
    st.markdown(f"<b>You:</b> {q}", unsafe_allow_html=True)
    st.markdown(f"<b>Assistant:</b> {a}", unsafe_allow_html=True)
