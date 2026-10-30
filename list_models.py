from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
models = client.models.list()
for m in sorted(models.data, key=lambda x: x.id):
    print(m.id)
