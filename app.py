from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from helper import create_vector_db


DB_PATH_VECTORSTORE = "vectorestore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


#Loading the model
def load_llm():

    load_dotenv() 
    os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

    return llm


#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_PATH_VECTORSTORE, embeddings, allow_dangerous_deserialization=True) 
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# Streamlit App
st.title("Q&A Assistant App")
st.write("Ask me anything about your document")

user_query = st.text_input("Enter your question:")

# Create sidebar for embedding section
with st.sidebar:
    st.write("Click the button below to make your app ready")
    if st.button("Documents Embedding"):
        with st.spinner('Embedding documents...'):  # Add spinner while embedding
            create_vector_db()
        st.write("Embedding completed! Proceed to query with your documents.")

if user_query:
    with st.spinner("Generating response..."):
        response = final_result(query=user_query)
        st.write(response['result'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["source_documents"]):
            st.write(doc.page_content)
            st.write("--------------------------------")