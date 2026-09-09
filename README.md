# DSPM Memory

**Compress multi-turn LLM conversations by 80%+ while guaranteeing every constraint and decision survives.**

`dspm-memory` is a training-free semantic memory compression package for conversations. It records typed semantic patches from turns and keeps critical constraint and decision patches protected under a fixed token budget.

## Install

```bash
pip install dspm-memory
```

## Quickstart

```python
from dspm import DSPMMemory

# Option A: use an OpenAI-compatible client
# llm_client = OpenAIClient(api_key="...", base_url="https://api.openai.com/v1")
# model = "gpt-4o-mini"

llm_client = None  # Replace with your own client in production.
memory = DSPMMemory(budget=250, llm_client=llm_client, model="gpt-4o-mini")

memory.add_turn("user", "Create an API that accepts a user id and returns JSON. Require auth tokens.")
memory.add_turn("assistant", "We will add an endpoint POST /v1/users and enforce bearer token authentication.")
memory.add_turn("user", "Only allow admin roles to list accounts.")
memory.add_turn("assistant", "I will add a decision that admin-only access is enforced in the route guard.")

context = memory.get_context(query="What constraints and decisions should the API remember?")
print(context)
print(memory.stats)
```

## Guarantee

The package converts multi-turn chats into semantic patches and compresses them under a budget. Constraint and decision patches are marked as critical and are retained structurally before all other patch types are considered. They may be trimmed to satisfy a hard budget, but they are not dropped unless the absolute last resort is reached.

## Results

| Budget | TRR | CRR |
|---|---:|---:|
| 250 | 82.84% | 100% |
| 400 | 72.39% | 100% |

More details are available in the included package docs and example.

## ArXiv Paper

A placeholder reference paper can be found at https://arxiv.org/abs/0000.00000.

## License

This project is licensed under the MIT License.
