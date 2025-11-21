"""
NLP Question-and-Answering System - Streamlit Web GUI
Author: [Your Name]
Matric No: [Your Matric Number]
Using Google Gemini API (FREE & RELIABLE!)
"""

import streamlit as st
import os
import string
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="NLP Q&A System",
    page_icon="🌟",
    layout="centered"
)

def preprocess_question(question):
    """
    Preprocess the input question by:
    - Converting to lowercase
    - Removing punctuation
    - Tokenizing
    """
    # Convert to lowercase
    question_lower = question.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    question_no_punct = question_lower.translate(translator)
    
    # Tokenize (split into words)
    tokens = question_no_punct.split()
    
    return tokens, question_no_punct

def query_llm(question, api_key, model_name):
    """
    Send question to Google Gemini API and get response
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel(model_name)
        
        # Generate response
        response = model.generate_content(question)
        
        answer = response.text
        return answer.strip(), None
        
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "invalid" in error_msg.lower():
            return None, "Invalid API key. Please check your Gemini API key."
        else:
            return None, f"Error: {error_msg}"

# Main app
def main():
    # Title and description
    st.title("🌟 NLP Question-and-Answering System")
    st.markdown("### Ask any question and get intelligent answers powered by AI")
    st.markdown("*Powered by Google Gemini - Free & Powerful!*")
    st.markdown("---")
    
    # Sidebar for API key and model selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Enter your Gemini API key. Get one FREE at https://makersuite.google.com/app/apikey"
        )
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Select Model")
        model_options = {
            "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 1.0 Pro": "gemini-1.0-pro"
        }
        
        selected_model_name = st.selectbox(
            "Choose AI Model",
            list(model_options.keys()),
            help="All models are FREE with Google Gemini!"
        )
        
        selected_model = model_options[selected_model_name]
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info(
            "This application uses Google's Gemini AI to answer your questions. "
            "Gemini is 100% FREE with generous rate limits!"
        )
        
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🔤 Text preprocessing
        - 🌟 Powered by Google AI
        - 📊 Token analysis
        - 🤖 Multiple AI models
        - 🆓 100% Free to use
        - ⚡ Fast responses
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Why Gemini?")
        st.success("""
        ✅ Completely FREE
        ✅ No organization needed
        ✅ 60 requests/minute
        ✅ Easy Google signup
        ✅ Reliable & stable
        ✅ State-of-the-art AI
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Get Free API Key")
        st.markdown("[Click here to get Gemini API key](https://makersuite.google.com/app/apikey)")
    
    # Main input area
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="E.g., What is the capital of France?",
        help="Type any question you'd like to ask"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("🚀 Get Answer", use_container_width=True, type="primary")
    
    # Process question when submitted
    if submit_button:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        elif not api_key:
            st.error("🔑 Please enter your Gemini API key in the sidebar.")
            st.info("👉 Get your FREE API key at: https://makersuite.google.com/app/apikey")
        else:
            # Show processing indicator
            with st.spinner("Processing your question..."):
                
                # Preprocessing section
                st.markdown("---")
                st.subheader("📝 Preprocessing Results")
                
                tokens, processed = preprocess_question(question)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Question:**")
                    st.info(question)
                
                with col2:
                    st.markdown("**Processed Question:**")
                    st.info(processed)
                
                st.markdown("**Tokens:**")
                st.code(" | ".join(tokens), language=None)
                st.caption(f"Total tokens: {len(tokens)}")
                
                # Query LLM
                st.markdown("---")
                st.subheader("🤖 AI Response")
                st.caption(f"Using model: {selected_model_name}")
                
                with st.spinner(f"Querying {selected_model_name}..."):
                    answer, error = query_llm(question, api_key, selected_model)
                
                if error:
                    st.error(f"❌ {error}")
                    if "Invalid" in error or "api_key" in error.lower():
                        st.warning("🔑 Your API key might be invalid. Please check it.")
                        st.info("Get a new key at: https://makersuite.google.com/app/apikey")
                else:
                    st.success("✅ Answer generated successfully!")
                    st.markdown("**Answer:**")
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #4285F4;
                    ">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show model info
                    with st.expander("ℹ️ Model Information"):
                        st.write(f"**Model Used:** {selected_model}")
                        st.write(f"**Provider:** Google Gemini")
                        st.write(f"**Speed:** ⚡ Fast")
                        st.write(f"**Cost:** 🆓 FREE")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>NLP Q&A System | Powered by Google Gemini 🌟</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()