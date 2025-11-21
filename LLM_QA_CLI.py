"""
NLP Question-and-Answering System - CLI Application
Author: [Your Name]
Matric No: [Your Matric Number]
Using Google Gemini API (FREE & RELIABLE!)
"""

import os
import string
import google.generativeai as genai

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

def query_llm(question, api_key=None):
    """
    Send question to Google Gemini API and get response
    Using Gemini 1.5 Flash - FREE and FAST!
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return "Error: API key not found. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(question)
        
        answer = response.text
        return answer.strip()
        
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}\n\nGet free API key at: https://makersuite.google.com/app/apikey"

def main():
    """
    Main CLI application loop
    """
    print("=" * 60)
    print("NLP Question-and-Answering System (CLI)")
    print("Powered by Google Gemini 🌟")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the session\n")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in environment variables.")
        api_key = input("Please enter your Gemini API key (or press Enter to exit): ").strip()
        if not api_key:
            print("Exiting...")
            return
    
    while True:
        # Get user input
        question = input("\nEnter your question: ").strip()
        
        # Check for exit command
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Q&A system. Goodbye!")
            break
        
        # Skip empty questions
        if not question:
            print("Please enter a valid question.")
            continue
        
        # Preprocess question
        print("\n" + "-" * 60)
        print("PREPROCESSING:")
        tokens, processed = preprocess_question(question)
        print(f"Original: {question}")
        print(f"Processed: {processed}")
        print(f"Tokens: {tokens}")
        
        # Query LLM
        print("\n" + "-" * 60)
        print("QUERYING GEMINI API...")
        answer = query_llm(question, api_key)
        
        # Display answer
        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("-" * 60)

if __name__ == "__main__":
    main()