import os
from openai import OpenAI

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY, "Set GROQ_API_KEY first"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)

print("Available models on Groq:")
for m in sorted(llm.models.list().data, key=lambda x: x.id):
    print(f"  {m.id}")
