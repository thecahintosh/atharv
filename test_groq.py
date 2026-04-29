# test_groq.py

from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
model   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    print("❌ API key not found")
    print("   Check your .env file")
else:
    print(f"✅ Key loaded  — starts with : {api_key[:8]}...")
    print(f"✅ Model loaded — using       : {model}")

print("\n🤖 Testing Groq connection...")

try:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role"   : "user",
                "content": "Say exactly this: Groq is working correctly."
            }
        ],
        max_tokens=20
    )

    reply = response.choices[0].message.content
    print(f"✅ Groq response: {reply}")
    print("\n🎉 Everything is working. You are ready to go.")

except Exception as e:
    print(f"❌ Error: {e}")