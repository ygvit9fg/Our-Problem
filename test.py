import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",  
    messages=[
        {"role": "system", "content": "Ты умный помощник."},
        {"role": "user", "content": "Привет! Скажи шутку."},
    ]
)

print(response.choices[0].message.content)  