import os
from openai import OpenAI
from dspm import DSPMMemory

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY, "Set key first: $env:GROQ_API_KEY = 'gsk_...'"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)
memory = DSPMMemory(budget=250, llm_client=llm, model="openai/gpt-oss-20b")

if os.path.exists("memory.json"):
    n = memory.load("memory.json")
    print(f"(restored {n} patches from previous chat)")

print("Commands: 'memory' = peek notebook | 'exit' = save & quit\n")

while True:
    user = input("You: ").strip()
    if user == "exit":
        memory.save("memory.json")
        print(f"(saved {len(memory.all_patches)} patches -> memory.json. See you next chat!)")
        break
    if user == "memory":
        print("\n--- DSPM notebook right now ---")
        print(memory.get_context(query="overview"))
        print("-------------------------------\n")
        continue

    ctx = memory.get_context(query=user)
    reply = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content":
             "You are a helpful assistant. MEMORY lines are your long-term knowledge: "
             "[CON]=constraint you must honor, [DEC]=decision made, "
             "[CODE]/[ENT]/[STR]/[EQ]=context. Answer using the memory plus the current message."},
            {"role": "user", "content": f"MEMORY:\n{ctx}\n\nMESSAGE: {user}"},
        ],
        temperature=0.3, max_tokens=1000,
    ).choices[0].message.content
    print("Bot:", reply, "\n")
    memory.add_turn("user", user)
    memory.add_turn("assistant", reply)
