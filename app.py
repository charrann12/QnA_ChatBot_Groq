import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os 

from dotenv import load_dotenv

load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "QnA_ChatBot_Groq"


## prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question,api_key,engine, temperature, max_tokens):
    
    llm = ChatGroq(groq_api_key = api_key,
                    model=engine,
                    temperature = temperature,
                    max_tokens = max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app 
st.title("Enhanced Q&A Chatbot with Groq")

## Sidebar for settings 
st.sidebar.title("Settings")

## for API Key
api_key = st.sidebar.text_input("Enter Your Groq API Key here", type="password")

## dropdown to select various Groq models

engine = st.sidebar.selectbox("Select a model",[ "llama-3.1-8b-instant","llama-3.3-70b-versatile",
"openai/gpt-oss-20b"])

## adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value = 150)

## Main interface for user input 

st.write("Go Ahead and ask anything")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
      st.warning("Please Enter your Groq API Key")  
    else:
        response = generate_response(user_input,api_key, engine, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide the query!")









