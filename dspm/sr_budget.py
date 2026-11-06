"""
dspm/sr_budget.py - v0.3.0 SR-Aware Budget Selection

Instead of user guessing a budget, DSPM finds the minimum budget
where Semantic Recall (SR) meets a target threshold.

Usage:
    mem = DSPMMemory(
        target_sr=0.8,           # I want 80% semantic recall
        llm_client=llm,
        model="gpt-4o-mini",
    )
    # DSPM automatically finds the minimum budget where SR >= 80%

Algorithm:
    1. Generate 5 factual questions from extracted patches (cached)
    2. Binary search for minimum budget where SR >= target
    3. Max 5 iterations (log2 of typical range)
"""
import json
import time
import logging

logger = logging.getLogger(__name__)


class SRAwareBudgetSelector:
    """Finds the minimum budget where Semantic Recall meets a target."""

    def __init__(self, llm_client, model, target_sr=0.8, max_iterations=5, questions=None):
        self.llm_client = llm_client
        self.model = model
        self.target_sr = target_sr
        self.max_iterations = max_iterations
        self._questions = questions
        self.last_sr = 0.0
        self.last_budget = 0

    def generate_questions(self, patches):
        """Generate 5 factual questions from extracted critical patches.
        
        Called once per conversation, then cached.
        Returns list of {"question": "...", "keywords": ["word1", "word2"]}
        """
        criticals = [p for p in patches if p.patch_type in {"constraint", "decision"}]
        if not criticals:
            return []

        # Take up to 10 critical patches as context for question generation
        patch_text = "\n".join(p.to_prompt_str() for p in criticals[:10])

        prompt = f"""Generate 5 factual questions about this conversation memory.
Each question should have a specific factual answer found in the text.
Focus on numbers, limits, thresholds, and specific values.

Return as a JSON array. Format:
[{{"question": "What is the JWT expiry time?", "keywords": ["30", "minute"]}}]

MEMORY:
{patch_text}

Return ONLY the JSON array, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{{"role": "user", "content": prompt}}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Parse JSON
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                logger.info(f"SR: Generated {{len(questions)}} questions")
                return questions[:5]  # Cap at 5
        except Exception as e:
            logger.warning(f"SR: Question generation failed: {{e}}")

        return []

    def _judge_answer(self, context, question):
        """Ask the judge LLM to answer a question using ONLY the compressed context."""
        prompt = (
            f"You are a fact-extraction engine. Answer the question using ONLY the context below. "
            f"If the answer is not in the context, say 'NOT FOUND'. "
            f"Be concise - just the answer, no explanation.\n\n"
            f"CONTEXT:\n{{context}}\n\n"
            f"QUESTION: {{question}}\n\n"
            f"ANSWER:"
        )
        for attempt in range(3):
            try:
                time.sleep(3)  # Rate limit protection
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{{"role": "user", "content": prompt}}],
                    temperature=0.0,
                    max_tokens=100,
                )
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    print(f"    Judge Q: '{question}' -> A: '{answer}'")
                    return answer
            except Exception as e:
                logger.warning(f"SR: Judge attempt {{attempt}} failed: {{e}}")
                time.sleep(3)
        return ""

    def _score_answer(self, answer, keywords):
        """Check if all keywords are present in the answer (case-insensitive)."""
        if not answer:
            return False
        answer_lower = answer.lower()
        if "not found" in answer_lower:
            return False
        return all(kw.lower() in answer_lower for kw in keywords)

    def probe_sr(self, context):
        """Ask judge LLM all questions using compressed context. Return SR score."""
        if not self._questions:
            return 0.0

        correct = 0
        for q in self._questions[:5]:
            answer = self._judge_answer(context, q["question"])
            is_correct = self._score_answer(answer, q.get("keywords", []))
            print(f"    Score: {'✓' if is_correct else '✗'} (keywords: {q.get('keywords', [])})")
            if is_correct:
                correct += 1

        sr = correct / len(self._questions[:5])
        logger.info(f"SR probe: {{correct}}/{{len(self._questions[:5])}} = {{sr:.1%}}")
        return sr

    def find_optimal_budget(self, compress_fn, floor, ceiling):
        """Binary search for minimum budget where SR >= target.

        Args:
            compress_fn: callable(budget) -> compressed_context_string
            floor: minimum budget (from min_budget_for_crr)
            ceiling: maximum budget (from ctx_window - reserve)

        Returns:
            (best_budget, best_sr)
        """
        if not self._questions:
            # No questions generated, return ceiling as fallback
            return ceiling, 0.0

        best_budget = ceiling
        best_sr = 0.0

        for iteration in range(self.max_iterations):
            mid = (floor + ceiling) // 2
            logger.info(f"SR search iter {{iteration}}: budget={{mid}} (floor={{floor}}, ceiling={{ceiling}})")

            context = compress_fn(mid)
            sr = self.probe_sr(context)

            if sr >= self.target_sr:
                best_budget = mid
                best_sr = sr
                ceiling = mid - 1  # Try lower
                logger.info(f"SR={{sr:.1%}} >= {{self.target_sr:.1%}}, trying lower")
            else:
                floor = mid + 1  # Need higher
                logger.info(f"SR={{sr:.1%}} < {{self.target_sr:.1%}}, trying higher")

            if floor > ceiling:
                break

        self.last_sr = best_sr
        self.last_budget = best_budget
        return best_budget, best_sr
