# models/model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_model(model_path):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=False,
    # )

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto"
    #         )
   
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
            )
    return tokenizer, model

def load_embeddings(embedding_path):
    return HuggingFaceEmbeddings(model_name=embedding_path)
