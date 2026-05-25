from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate


class ExpRecent(BaseMethod):
    name = "ExpRecent"

    def __init__(
        self,
        top_k=3,
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_new_tokens=1024,
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.generation_model_name = generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

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
            "Use the retrieved recent history as reference, but solve the current problem independently.\n"
            "Return your final answer clearly.\n\n"
            f"Retrieved memory:\n{context_text}\n\n"
            f"Current problem:\n{prompt}\n"
        )

    def retrieve_memory(self, task, **kwargs):
        task_rows = [row for row in self.memory if row.get("task") == task.name]
        if self.top_k <= 0:
            return []
        return task_rows[-self.top_k :]

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        generation_prompt = self._build_generation_prompt(prompt, retrieved)
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return
        enriched = {"task": task.name, **record}
        self.memory.append(enriched)
        enriched["memory"] = self.build_task_memory_snapshot(task.name)

    def reset_task_state(self, task):
        return None
