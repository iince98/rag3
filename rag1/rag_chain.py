from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
# from langchain.chains.combine_documents import StuffDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from config import RETRIVAL_K

def build_qa_chain(model, tokenizer, vector_db):
    # Prompt template
    prompt = PromptTemplate(
        template="""
Beantworte die Frage des Nutzers unter Verwendung des folgenden Kontexts, **wenn dieser relevant ist**. 
Die Informationen stammen aus mehreren Dokumentenausschnitten. Wenn der Kontext nicht hilft, beantworte die Frage **nicht** aus eigenem Wissen.

Jeder Ausschnitt ist durch "---" getrennt.

Kontext:
{context}

Frage:
{question}

Antwort:
""",
        input_variables=["context", "question"]
    )

    # Hugging Face generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Chain that merges documents with separators
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_separator="\n---\n"  #inserts the separator
    )

    # Retriever with k
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIVAL_K})

    # Final RetrievalQA chain
    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_docs_chain,
        return_source_documents=True
    )