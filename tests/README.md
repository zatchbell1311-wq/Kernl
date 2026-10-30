# DSPM Tests

## Setup
```bash
pip install dspm-memory
python -m spacy download en_core_web_sm
```

Set your API key (Groq recommended — free tier available at console.groq.com):
```powershell
$env:GROQ_API_KEY = "gsk_..."
```

## Test 1 — Quick CRR verification (no LLM needed for offline part)
```bash
python tests/verify_crr.py
```
Expected: `PASS -- CRR = 100%`

## Test 2 — Offline universal budget sweep (no LLM calls)
```bash
python tests/test_universal_budgets.py
```
Expected: `ALL PASS -- CRR = 100% at every budget from 59 to 5000 tokens`

## Test 3 — Small live test (10 turns, needs API key)
```bash
python tests/small_live_test_budgets.py
```
Expected results:
| Budget | CRR | TRR |
|--------|-----|-----|
| 60 | 100% | ~71% |
| 100 | 100% | ~55% |
| 250 | 100% | ~-4% |
| 400 | 100% | ~-28% |

## Test 4 — Large live test (24 turns, 3 revisions, needs API key)
```bash
python tests/large_live_test_budgets.py
```
Expected results:
| Budget | CRR | TRR |
|--------|-----|-----|
| 60 | 100% | ~93% |
| 100 | 100% | ~81% |
| 250 | 100% | ~59% |
| 400 | 100% | ~36% |
| 800 | 100% | ~0% |

Note: TRR varies by conversation length.
Recommended budget: 250-400 for best readability + compression balance.
