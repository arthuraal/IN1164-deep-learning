import torch

from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, \
    DataCollatorForLanguageModeling


class LlamaModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = 0

