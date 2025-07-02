import streamlit as st
from assistant import get_llm
from transformers import AutoTokenizer

st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
    }
    .big-font {
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo and title
st.image("https://upload.wikimedia.org/wikipedia/commons/8/8f/OpenAI_Logo.svg", width=70)
st.title("AI Writing Assistant")
st.caption("Powered by IBM Watsonx · Ask academic, legal, creative, or professional questions")

llm = get_llm()
MAX_TOKENS = 312  # This must match the limit in assistant.py

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

with st.sidebar:
    st.header("🛠 Settings")
    st.markdown("Customize your interaction")
    st.info("Using model: Mistral Large")
    st.markdown("\nTry prompts like:")
    st.code("Write a cover letter for a data science internship")
    st.code("Explain Newton's laws in simple terms")

st.markdown("<div class='big-font'> Enter your writing prompt below:</div>", unsafe_allow_html=True)
query = st.text_area("", height=120, placeholder="e.g. Write a story about a girl who finds a time-travel watch...")

if query:
    with st.spinner("Generating..."):
        response = llm.invoke(query)
        st.success("✅ Response ready")
        st.markdown("**Output:**")
        st.code(response, language="markdown")

        # Token overflow warning
        num_tokens = len(tokenizer.encode(response))
        if num_tokens >= int(MAX_TOKENS * 0.95):
            st.warning("⚠️ Response may have been cut off due to token limit.")
