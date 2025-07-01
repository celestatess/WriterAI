import streamlit as st
from assistant import get_llm
from transformers import AutoTokenizer
st.set_page_config(page_title="AI Writing Assistant", layout="centered")
st.title("AI Writing Assistant")

llm = get_llm()
MAX_TOKENS = 312  #limit

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
query = st.text_input("Ask me anything:")

if query:
    with st.spinner("Thinking..."):
        response = llm.invoke(query)
        st.markdown(f"**Answer:** {response}")

        num_tokens = len(tokenizer.encode(response))
        if num_tokens >= int(MAX_TOKENS * 0.95):
            st.warning("⚠️ This response may have been cut off due to token limits.")
