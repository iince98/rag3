import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_model(model_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        "iince98/mistral-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("iince98/mistral-7b-instruct")
    return tokenizer, model

def load_embeddings(embedding_path):
    return HuggingFaceEmbeddings(model_name=embedding_path)
