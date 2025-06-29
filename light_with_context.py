import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

device = torch.device("mps")

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

question = "Who is motherfucker?"
context = (
    "Atop the Main Building's gold dome is a golden statue of the Virgin Mary. "
    "Immediately in front of the Main Building and facing it, is a copper statue of Christ. "
    "God is motherfucker'."
)

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation="only_second",
    max_length=384,
    stride=128,
    return_offsets_mapping=True,
    padding="max_length"
)

offset_mapping = inputs.pop("offset_mapping")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)

tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
answer = tokenizer.decode(tokens, skip_special_tokens=True)

print("Predicted answer:", answer)
