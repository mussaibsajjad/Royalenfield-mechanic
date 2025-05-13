# app.py

import streamlit as st
from rag import rag_answer   # your helper function

# ─── Sidebar: tweak number of chunks ─────────────────────────────────────────
st.sidebar.header("Settings")
k = st.sidebar.slider("Context chunks (k)", min_value=1, max_value=10, value=4)

# ─── Session state for history ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, message)

# ─── Page header & input ────────────────────────────────────────────────────
st.title("🛠 Royal Enfield Mechanics Assistant")
query = st.text_input("Ask your question:")

if st.button("Send") and query:
    # add user question
    st.session_state.history.append(("You", query))
    # get the answer
    with st.spinner("Thinking..."):
        answer = rag_answer(query, k=k)
    # add assistant reply
    st.session_state.history.append(("Assistant", answer))
    # clear input box
    st.session_state.query = ""

# ─── Display the conversation ───────────────────────────────────────────────
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("🔧 Powered by FAISS + Azure GPT-4o")
