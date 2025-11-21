"""
NLP Question & Answering System - Streamlit Web App
Author: Afolabi Ayomide
"""

import streamlit as st
import os
import string
import google.generativeai as genai

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NLP Q&A System",
    page_icon="🌟",
    layout="centered"
)

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_question(question):
    """
    Converts question to lowercase, removes punctuation, and tokenizes.
    """
    question_lower = question.lower()
    translator = str.maketrans("", "", string.punctuation)
    question_no_punct = question_lower.translate(translator)
    tokens = question_no_punct.split()
    return tokens, question_no_punct

# -----------------------------
# GEMINI LLM QUERY FUNCTION
# -----------------------------
def query_llm(question, api_key, model_name):
    """
    Sends question to Google Gemini API using the selected model.
    Returns (answer, error).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(question)
        return response.text.strip(), None

    except Exception as e:
        msg = str(e)
        if "api_key" in msg.lower() or "invalid" in msg.lower():
            return None, "Invalid API Key. Please verify your Gemini API key."
        if "404" in msg or "not supported" in msg.lower():
            return None, "Model not found or unsupported. Please select a valid Gemini model."
        return None, f"Unexpected Error: {msg}"

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("🌟 NLP Question-and-Answering System")
    st.markdown("### Ask a question and get an AI-generated answer.")
    st.markdown("---")

    # -----------------------------
    # SIDEBAR CONFIG
    # -----------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = st.text_input(
            "🔐 Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Get your FREE Gemini API key at: https://makersuite.google.com/app/apikey"
        )

        st.markdown("---")
        st.subheader("🤖 Select AI Model")

        # ✅ Only working models for generateContent
        model_options = {
            "Gemini Flash (Latest)": "models/gemini-flash-latest",
            "Gemini Pro (Latest)": "models/gemini-pro-latest"
        }

        model_label = st.selectbox("Model", list(model_options.keys()))
        model_name = model_options[model_label]

        st.markdown("---")
        st.info("Uses Google's Gemini API — fast, free & powerful.")

        st.markdown("### Features")
        st.markdown("""
        - 🔤 Text preprocessing  
        - 🌐 Multiple Gemini models  
        - ⚡ Fast AI responses  
        - 🆓 100% Free  
        """)

    # -----------------------------
    # QUESTION INPUT
    # -----------------------------
    st.subheader("📝 Enter Your Question")
    question = st.text_area(
        "Type your question:",
        height=120,
        placeholder="E.g., What is natural language processing?",
    )

    submit = st.button("🚀 Get Answer", use_container_width=True)

    # -----------------------------
    # PROCESSING
    # -----------------------------
    if submit:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
            return
        if not api_key.strip():
            st.error("🔑 Please add your Gemini API key.")
            return

        st.markdown("---")
        st.subheader("🔍 Preprocessing Results")
        tokens, processed = preprocess_question(question)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Question:**")
            st.info(question)
        with col2:
            st.markdown("**Processed Question:**")
            st.info(processed)

        st.markdown("**Tokens:**")
        st.code(" | ".join(tokens))
        st.caption(f"Total tokens: {len(tokens)}")

        # -----------------------------
        # LLM QUERY
        # -----------------------------
        st.markdown("---")
        st.subheader("🤖 AI Response")
        st.caption(f"Model used: {model_label}")

        with st.spinner("Thinking..."):
            answer, error = query_llm(question, api_key, model_name)

        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Response generated!")
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fc;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 5px solid #4285F4;
                ">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("ℹ️ Model Information"):
                st.write(f"**Model ID:** {model_name}")
                st.write("**Provider:** Google Gemini")
                st.write("**Cost:** Free")
                st.write("**Latency:** Low / Fast")

    # -----------------------------
    # FOOTER
    # -----------------------------
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:gray;'>NLP Q&A System | Powered by Google Gemini 🌟</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
