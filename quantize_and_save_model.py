from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import  config

def quantize_and_save_model(model_name_or_path, save_path):
    # Define quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Save both model and tokenizer to disk
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Quantized model saved to {save_path}")

quantize_and_save_model (config.MODEL_PATH, config.QUANTIZED_MODEL_PATH)
