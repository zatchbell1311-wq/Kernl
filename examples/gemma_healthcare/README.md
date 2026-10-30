# DSPM with local Gemma - Healthcare Dialogue Test

Contributed by **Piotr Chlebek** (patent holder, healthcare dialogue processing).

## What this tests

96-turn home care support dialogue (caregiver to agent check-in call),
compressed from raw transcript to 1024 tokens using DSPM with a locally-hosted
Gemma model via vLLM.

## Setup

Start vLLM with Gemma (Piotr - confirm exact model tag in your PR):

    vllm serve <your-gemma-model-tag> --max-model-len 8192

Install DSPM:

    pip install dspm-memory

## Run

Add your run_test.py here (Piotr's PR).

## Results

- CRR = 100%
- 96 turns compressed to 1024 tokens
- All load-bearing medical facts preserved verbatim
