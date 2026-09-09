"""Runnable example for the DSPM memory package.

This example shows how to wire an OpenAI-compatible LLM client into the
memory layer. Groq or Ollama-compatible clients can be swapped in the same
way by changing the base_url and model values.
"""

from dspm import DSPMMemory


def main() -> None:
    # Replace this with an OpenAI-compatible client:
    # from openai import OpenAI
    # client = OpenAI(api_key="sk-...")
    # Or swap to Groq and Ollama via a different provider client.
    llm_client = None

    memory = DSPMMemory(budget=250, llm_client=llm_client, model="gpt-4o-mini")

    # Four realistic turns of an API design conversation.
    memory.add_turn("user", "Design a REST API for user profile management.")
    memory.add_turn("assistant", "Use a POST /v1/profiles endpoint and require bearer token auth.")
    memory.add_turn("user", "Only admins may delete a profile.")
    memory.add_turn("assistant", "Constraint: admin role required for DELETE /v1/profiles. Decision: client must send an account id.")

    context = memory.get_context(query="What must the API remember?")
    print(context)
    print(memory.stats)


if __name__ == "__main__":
    main()
