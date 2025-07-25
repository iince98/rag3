# data/vectorstore.py

import os
from langchain_community.vectorstores import Chroma

def create_vector_database(chunks, embeddings, persist_directory):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_db.persist()
        return vector_db
