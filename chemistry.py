import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

with open("chemistry_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

def chunk_text(text, max_len=400, stride=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len - stride):
        chunk = " ".join(words[i:i + max_len])
        chunks.append(chunk)
        if i + max_len >= len(words):
            break
    return chunks

chunks = chunk_text(full_text)

vectorizer = TfidfVectorizer().fit(chunks)
chunk_vectors = vectorizer.transform(chunks)

def find_best_chunk(question):
    question_vec = vectorizer.transform([question])
    scores = np.dot(chunk_vectors, question_vec.T).toarray().flatten()
    best_idx = np.argmax(scores)
    return chunks[best_idx]

def answer_question(question, context):
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
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    if end_idx < start_idx:
        end_idx = start_idx

    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    return answer.strip()

while True:
    question = input("Задай вопрос (или 'выход'): ")
    if question.lower() == "выход":
        break
    context = find_best_chunk(question)
    print("\n[Выбранный чанк]:\n", context[:300] + "...\n")
    answer = answer_question(question, context)
    print("Ответ:", answer or "[Пусто]\n")
