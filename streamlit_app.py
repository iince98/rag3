# app.py
# For DEMO
import streamlit as st
from config import MODEL_PATH, EMBEDDING_MODEL_PATH, DATA_PATH, PERSIST_DIRECTORY
from models.model_loader import load_model, load_embeddings
from data.document_loader import load_documents
from data.splitter import split_documents
from data.vectorstore import create_vector_database
from rag1.rag_chain import build_qa_chain

@st.cache_resource
def initialize_rag_pipeline():
    tokenizer, model = load_model(MODEL_PATH)
    embeddings = load_embeddings(EMBEDDING_MODEL_PATH)

    documents = load_documents(DATA_PATH)
    if not documents:
        return None, "No documents found."

    chunks = split_documents(documents)
    vector_db = create_vector_database(chunks, embeddings, PERSIST_DIRECTORY)
    qa_chain = build_qa_chain(model, tokenizer, vector_db)
    return qa_chain, None

def main():
    st.set_page_config(page_title="RAG QA System", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (RAG) QA System1")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    qa_chain, error = initialize_rag_pipeline()
    if error:
        st.error(error)
        return

    # Inject CSS to fix the form at the bottom and make upper part scrollable
    st.markdown(
        """
        <style>

        /* Target the overall message block for user and assistant */
        .stChatMessage.user {
            margin-left: 0rem !important;
        }

        .stChatMessage.assistant {
            margin-left: 2rem !important;
        }

        /* Optional: Tweak the message content style if needed */
        .user-message, .assistant-message {
            display: inline-block;
            width: 100%;
        }
        .user-message {
        margin-left: 0;
        padding-left: 1rem;
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        }

        .assistant-message {
        margin-left: 2rem;
        padding-left: 1rem;
        background-color: #fff8dc;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        }
        .main-container {
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .chat-scroll-area {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
            margin-bottom: 1rem;
        }
        .stForm {
            position: sticky;
            bottom: 0;
            background-color: white;
            padding-top: 10px;
            padding-bottom: 10px;
            z-index: 999;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🧠 Chat History")
    for _, (q, a, _) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(f"""
            <div class="user-message"><strong>Q:</strong> {q}</div>
            """, unsafe_allow_html=True)

        with st.chat_message("assistant"):
            st.markdown(f"""
            <div class="assistant-message"><strong>A:</strong> {a.split('Antwort:')[-1]}</div>
            """, unsafe_allow_html=True)

    # Latest sources shown here
    if st.session_state.chat_history:
        _, _, latest_sources = st.session_state.chat_history[-1]
        st.subheader("📄 Sources for Latest Answer")
        with st.expander("📄 Sources"):
            if latest_sources:
                for j, doc in enumerate(latest_sources):
                    meta = getattr(doc, "metadata", {})
                    content = getattr(doc, "page_content", "")
                    st.markdown(f"**Source {j+1}:** {meta.get('source', 'N/A')} | Page {meta.get('page', 'N/A')}")
                    st.text(content[:1000] + "..." if content else "_No content available_")
                    st.markdown("---")
            else:
                st.markdown("_No sources returned._")

    # Close scrollable area
    st.markdown('</div>', unsafe_allow_html=True)

    # Form at the bottom (always visible)
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Ask")

    # Take the chat histor into account
    # if submitted and query.strip():
    #     with st.spinner("Retrieving answer..."):
    #         # Build conversational context
    #         conversation_context = ""
    #         for past_q, past_a, _ in st.session_state.chat_history[-3:]:  # Limit to last 3 exchanges for brevity
    #             conversation_context += f"Q: {past_q}\nA: {past_a}\n"
    #         conversation_context += f"Q: {query}\nA:"

    #         # Send full conversational prompt to the RAG chain
    #         result = qa_chain.invoke({"query": conversation_context})
    #         answer = result['result']
    #         sources = result.get('source_documents', [])

    #         st.session_state.chat_history.append((query, answer, sources))
    #         st.rerun()
            
    
    
    if submitted and query.strip():
        with st.spinner("Retrieving answer..."):
            result = qa_chain.invoke({"query": query})
            answer = result['result']
            sources = result.get('source_documents', [])
            st.session_state.chat_history.append((query, answer, sources))

        # Force rerun to update scrollable section
        st.rerun()

    # Close outer container
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()