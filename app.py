import os
import socket
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
ONLINE_MODEL = "gpt-oss-20b"
OFFLINE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def internet_available(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def online_response(prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=ONLINE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]


def offline_response(prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(OFFLINE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(OFFLINE_MODEL, device_map="auto", torch_dtype=torch.float16)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def speak(text: str, lang="ru"):
    tts = gTTS(text=text, lang=lang)
    tts.save("reply.mp3")
    os.system("mpg123 reply.mp3")  # Linux/macOS. Для Windows: os.system("start reply.mp3")


if __name__ == "__main__":
    print("Гибридный ассистент запущен!")
    while True:
        user_input = input("Ты: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            break

        if internet_available():
            print("💻 Онлайн режим (gpt-oss)...")
            reply = online_response(user_input)
        else:
            print("📴 Оффлайн режим (Mistral 7B)...")
            reply = offline_response(user_input)

        print("🤖 Ассистент:", reply)
        speak(reply)
