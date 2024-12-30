import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager
from chatbot import ChatBotManager

# Function to display the PDF
# Function to display the PDF of a given file
def displayPDF(file):
    # Reading the uploaded file
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

# Intialize session_state variable 
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Chat with PDF app",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path 
    st.image("chatbot.png", use_container_width=True)
    st.markdown("### Your personal Document Assistant")
    st.markdown("---")

    # Navigation Menu
    menu =["🏠 Home", "🤖 Chatbot",]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("Chat with PDF App")
    st.markdown("""
    Welcome to **Document Buddy App**! 🚀

    **Built using Open Source Stack (Llama 3.2, BGE Embeddings, and Qdrant running locally within a Docker Container.)**

    - **Upload Documents**: Easily upload your PDF documents.
    - **Summarize**: Get concise summaries of your documents.
    - **Chat**: Interact with your documents through the intelligent chatbot.

    Enhance your document managment experience!
    """)
# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface (Llama 3.2 RAG 🦙)")
    st.markdown("---")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File uploader and preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            # Diplay file name and size
            st.markdown(f"File Name: {uploaded_file.name}")
            st.markdown(f"File Size: {uploaded_file.size} bytes")

            # Diplay PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview")
            displayPDF(uploaded_file)

            # Save the uploaded file to a temporary location\
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Store the temp_pdf_path in session_state
            st.session_state['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("Create Embeddings")
        if create_embeddings:
            if st.session_state['temp_pdf_path'] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    # Intialize the EmbeddingsManager
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db"
                    )

                    with st.spinner("Generating Embeddings..."):
                        # Create Embeddings
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)
                    st.success(result)

                    # Intialize the ChatBotManager after embeddings are created
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatBotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama3.2:3b",
                            llm_temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    # Column 3: ChatBot Interface
    with col3:
        st.header("💬 Chat with Document")

        if st.session_state['chatbot_manager'] is None:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting!")
        else:
            # Create a container for the chat interface
            chat_container = st.container()

            # Display existing messages inside the container
            with chat_container:
                for msg in st.session_state['messages']:
                    if msg['role'] == "user":
                        st.chat_message("user").markdown(msg['content'])
                    elif msg['role'] == "assistant":
                        st.chat_message("assistant").markdown(msg['content'])

            # Keep chat input at the bottom for continuous interaction
            user_input = st.chat_input("Type your message here...")

            if user_input:
                # Display user message in real-time
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.chat_message("user").markdown(user_input)

                with st.spinner("Responding..."):
                    try:
                        # Get the chatbot response using the ChatBotManager
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"

                # Display assistant's response in real-time
                st.session_state['messages'].append({"role": "assistant", "content": answer})
                st.chat_message("assistant").markdown(answer)


