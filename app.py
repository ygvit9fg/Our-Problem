import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


LOCAL_MODEL = "gpt2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Загрузка локальной модели {LOCAL_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

def ask_online(prompt: str) -> str:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты умный помощник."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def ask_offline(prompt: str) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


if __name__ == "__main__":
    question = "Привет, расскажи интересный факт про космос!"

    mode = input("Выбери режим (online/offline): ").strip().lower()

    if mode == "online":
        print("🌐 Онлайн режим:")
        print(ask_online(question))

    elif mode == "offline":
        print("💻 Оффлайн режим:")
        print(ask_offline(question))

    else:
        print("⚠️ Неверный выбор! Напиши 'online' или 'offline'.")
