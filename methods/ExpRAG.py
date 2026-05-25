import math

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    transformers_logging = None


class ExpRAG(BaseMethod):
    name = "ExpRAG"

    def __init__(
        self,
        top_k=3,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_new_tokens=1024,
    ):
        super().__init__()
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._embedder = None
        self._memory_vectors = []

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "ExpRAG requires sentence-transformers. "
                "Please install it first: pip install sentence-transformers"
            ) from exc
        self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _encode(self, texts):
        if not texts:
            return []
        embedder = self._load_embedder()
        vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(vec) for vec in vectors]

    def _build_generation_prompt(self, prompt, retrieved):
        context_lines = []
        for idx, item in enumerate(retrieved, start=1):
            context_lines.append(
                f"[Memory {idx}]\n"
                f"Question: {item.get('question', '')}\n"
                f"Model Output: {item.get('model_output', '')}\n"
                f"Feedback: {item.get('feedback', '')}"
            )
        context_text = "\n\n".join(context_lines) if context_lines else "No retrieved memory."
        return (
            "You are solving a math problem.\n"
            "Use the retrieved history as reference, but solve the current problem independently.\n"
            "Return your final answer clearly.\n\n"
            f"Retrieved memory:\n{context_text}\n\n"
            f"Current problem:\n{prompt}\n"
        )

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve_memory(self, task, **kwargs):
        if not self.memory:
            return []
        query = str(kwargs.get("query", ""))
        if not query.strip():
            return self.memory[-self.top_k :]
        query_vec = self._encode([query])[0]
        scored = []
        for idx, record in enumerate(self.memory):
            rec_task = record.get("task")
            if rec_task and rec_task != task.name:
                continue
            score = self._cosine(query_vec, self._memory_vectors[idx])
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[: self.top_k]]

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        generation_prompt = self._build_generation_prompt(prompt, retrieved)
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def test_build_solver_prompt(self, task, entry, retrieved, inputs):
        # ExpRAG has no separate cheatsheet/rules: memory == retrieved
        # past Q/A. The retrieval is already per-question; we just build
        # the same generation prompt used in training.
        prompt = task.build_prompt(entry)
        return self._build_generation_prompt(prompt, retrieved or [])

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return
        enriched = {"task": task.name, **record}
        self.memory.append(enriched)
        enriched["memory"] = self.build_task_memory_snapshot(task.name)
        text = (
            f"Q: {enriched.get('question', '')}\n"
            f"A: {enriched.get('model_output', '')}\n"
            f"Feedback: {enriched.get('feedback', '')}"
        )
        self._memory_vectors.extend(self._encode([text]))

    def reset_task_state(self, task):
        # Keep memory across the stream of one dataset.
        return None

    def restore_state_from_memory(self, task):
        """Re-embed all loaded memory records so retrieve_memory works after
        loading from a saved *_memory.jsonl via test_only.py."""
        texts = []
        for record in self.memory:
            if record.get("task") and record.get("task") != task.name:
                texts.append("")
                continue
            texts.append(
                f"Q: {record.get('question', '')}\n"
                # f"A: {record.get('model_output', '')}\n"
                # f"Feedback: {record.get('feedback', '')}"
            )
        self._memory_vectors = self._encode(texts) if any(texts) else []
        print(
            f"[ExpRAG] restored {len(self.memory)} memory records "
            f"for task={task.name}, vectors={len(self._memory_vectors)}"
        )

