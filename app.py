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
    torch_dtype=torch.float32,   # лучше для стабильности
).to(device)  # 🚀 явно грузим на CPU или GPU


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
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔎 Отладка: показываем весь текст
    print("\n[DEBUG] Сырой ответ модели:")
    print(generated_text)

    clean_answer = generated_text[len(prompt):].strip()
    print("\n[OFFLINE] Ответ модели:")
    print(clean_answer)
    return clean_answer




if __name__ == "__main__":
    # Тестируем генерацию напрямую
    inputs = tokenizer("Привет! Расскажи про Италию.", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("\n[TEST] Генерация напрямую:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    mode = input("Выбери режим (online/offline): ").strip().lower()

    while True:
        question = input("\n❓ Вопрос (или 'exit' для выхода): ")

        if question.lower() in ["exit", "quit", "выход"]:
            print("👋 Выход из программы.")
            break

        if mode == "online":
            print("\n🌐 Онлайн режим:")
            print(ask_online(question))

        elif mode == "offline":
            print("\n💻 Оффлайн режим:")
            print(ask_offline(question))

        else:
            print("⚠️ Неверный выбор! Напиши 'online' или 'offline'.")
            break
