import os
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Загрузка локальной модели {LOCAL_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def ask_online(prompt: str) -> str:

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты умный помощник."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Онлайн ошибка] {e}")
        return None

def ask_offline(prompt: str) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def ask(prompt: str) -> str:

    answer = ask_online(prompt)
    if answer is None:
        print("⚡ Переключаемся в оффлайн-режим...")
        answer = ask_offline(prompt)
    return answer



if __name__ == "__main__":
    question = "Привет, расскажи интересный факт про космос!"
    print("Вопрос:", question)
    answer = ask(question)
    print("Ответ:", answer)
