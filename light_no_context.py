import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("mps") if torch.has_mps else torch.device("cpu")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def answer_question(question: str):
    input_text = f"question: {question} context: "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    while True:
        q = input("Задай вопрос (или 'выход'): ").strip()
        if q.lower() in ("выход", "exit"):
            break
        a = answer_question(q)
        print("Ответ:", a)
