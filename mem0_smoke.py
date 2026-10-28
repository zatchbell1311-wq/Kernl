from mem0 import Memory

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.0,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "dspm_mem0_test",
            "embedding_model_dims": 768,
            "path": "./qdrant_local_db",
        },
    },
}

m = Memory.from_config(config)
m.add("User prefers dark mode and works in Python.", user_id="test_user")
result = m.search("What does the user prefer?", filters={"user_id": "test_user"})
print(result)
