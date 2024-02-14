import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from urllib.parse import urlparse, urlunparse

load_dotenv()


def format_url(url):
    # Check if the URL has a scheme, if not, assume 'https' and prepend '//' for proper parsing
    if not urlparse(url).scheme:
        url = '//' + url

    # Parse the URL
    parsed_url = urlparse(url, scheme='https')

    # Scheme handling: Ensure the scheme is 'https'
    scheme = 'https'

    # Netloc handling: Directly use the parsed netloc, avoiding adding 'www.' if not appropriate
    netloc = parsed_url.netloc

    # Special handling to avoid adding 'www.' to domains that are already subdomains or include 'www.'
    if not netloc.startswith('www.') and netloc.count('.') == 1:
        netloc = 'www.' + netloc

    # Reconstruct the URL, ensuring only two slashes are used between the scheme and netloc
    formatted_url = urlunparse((scheme, netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))

    return formatted_url

def get_vector_store_from_url(url):
    # get the text in document format
    reformatted_url = format_url(url)
    loader = WebBaseLoader(reformatted_url)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vector store from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    # Create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="👽")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("⬅️Please enter a URL in the sidebar")

else:
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = \
        [
            AIMessage(content="Document uploaded! What's your question?"),
        ]
    
    if "vector_state" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)

    # user input
    user_query = st.chat_input("Type your question here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)