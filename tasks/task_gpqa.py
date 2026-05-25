import hashlib
import random
import re

from datasets import load_dataset

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import extract_answer_tag


def _normalize_text(text):
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_choice_label(text):
    text = _normalize_text(text)
    if not text:
        return ""
    ch = text[0]
    if ch in {"a", "b", "c", "d"}:
        return ch.upper()
    return ""


class GPQATask(BaseTask):
    name = "GPQA"
    prompt_key = "gpqa"

    def __init__(
        self,
        prompt_file=None,
        hf_path="Idavidrein/gpqa",
        split="diamond",
        hf_split="train",
    ):
        self.prompt_file = prompt_file
        self.hf_path = hf_path
        self.split = str(split or "diamond").strip().lower()
        self.hf_split = hf_split
        self._entries = None

    def _resolve_subset_and_mode(self):
        # Canonical setup:
        # - test  : gpqa_diamond
        # - train : gpqa_main \ gpqa_diamond
        split_alias = self.split
        if split_alias in {"diamond", "test", "testonly", "eval"}:
            return "gpqa_diamond", "direct"
        if split_alias in {
            "train",
            "main_wo_diamond",
            "main_without_diamond",
            "main_minus_diamond",
            "main_non_diamond",
        }:
            return "gpqa_main", "exclude_diamond"
        if split_alias in {"main", "gpqa_main"}:
            return "gpqa_main", "direct"
        if split_alias in {"diamond_raw", "gpqa_diamond"}:
            return "gpqa_diamond", "direct"
        raise ValueError(
            "Unsupported GPQA split. Use one of: "
            "diamond/test, train/main_wo_diamond, main."
        )

    def _question(self, item):
        return item.get("Question", "") or item.get("question", "")

    def _correct_answer_text(self, item):
        return item.get("Correct Answer", "") or item.get("correct_answer", "")

    def _incorrect_answers(self, item):
        incorrect = []
        for key in (
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ):
            if key in item and str(item.get(key, "")).strip():
                incorrect.append(str(item[key]).strip())
        return incorrect

    def _entry_signature(self, item):
        question = _normalize_text(self._question(item))
        answer = _normalize_text(self._correct_answer_text(item))
        return f"{question}||{answer}"

    def _stable_rng(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        return random.Random(seed)

    def _build_choices(self, item):
        labels = ["A", "B", "C", "D"]
        correct_text = str(self._correct_answer_text(item)).strip()
        wrong_texts = self._incorrect_answers(item)

        # Fallback: if dataset already stores lettered choices.
        explicit = []
        for label in labels:
            if label in item and str(item.get(label, "")).strip():
                explicit.append((label, str(item[label]).strip()))
        if len(explicit) == 4:
            correct_raw = (
                item.get("Correct Choice")
                or item.get("correct_choice")
                or item.get("answer")
                or ""
            )
            gold_label = _normalize_choice_label(correct_raw)
            return explicit, gold_label, dict(explicit).get(gold_label, "")

        pool = [(correct_text, True)] + [(w, False) for w in wrong_texts]
        # Ensure we always have exactly 4 candidates.
        dedup = []
        seen = set()
        for text, is_correct in pool:
            key = _normalize_text(text)
            if not key or key in seen:
                continue
            dedup.append((text, is_correct))
            seen.add(key)
        if len(dedup) < 4:
            raise RuntimeError(
                "GPQA sample does not have enough distinct answer options."
            )
        dedup = dedup[:4]

        rng = self._stable_rng(self._question(item))
        rng.shuffle(dedup)
        options = []
        gold_label = ""
        gold_text = ""
        for idx, (text, is_correct) in enumerate(dedup):
            label = labels[idx]
            options.append((label, text))
            if is_correct:
                gold_label = label
                gold_text = text
        return options, gold_label, gold_text

    def _load(self):
        if self._entries is not None:
            return

        subset, mode = self._resolve_subset_and_mode()
        primary = load_dataset(self.hf_path, subset, split=self.hf_split)
        rows = [dict(item) for item in primary]

        if mode == "exclude_diamond":
            diamond = load_dataset(self.hf_path, "gpqa_diamond", split=self.hf_split)
            diamond_signatures = {self._entry_signature(dict(item)) for item in diamond}
            rows = [
                item
                for item in rows
                if self._entry_signature(item) not in diamond_signatures
            ]

        entries = []
        skipped_invalid_choices = 0
        for idx, item in enumerate(rows):
            qid = str(item.get("id") or item.get("ID") or idx)
            try:
                options, gold_label, gold_text = self._build_choices(item)
            except RuntimeError as exc:
                if "distinct answer options" not in str(exc):
                    raise
                skipped_invalid_choices += 1
                continue
            prepared = dict(item)
            prepared["_qid"] = qid
            prepared["_options"] = options
            prepared["_gold_label"] = gold_label
            prepared["_gold_text"] = gold_text
            entries.append(prepared)
        if skipped_invalid_choices:
            print(
                f"[GPQA] skipped {skipped_invalid_choices} samples with "
                "insufficient distinct answer options."
            )
        self._entries = entries

    def iter_entries(self):
        self._load()
        for item in self._entries:
            yield item

    def total_entries(self):
        self._load()
        return len(self._entries)

    def build_prompt(self, entry):
        choices_text = "\n".join([f"{k}. {v}" for k, v in entry.get("_options", [])])
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
            choices=choices_text,
        )

    def build_inputs(self, entry):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "choices": entry.get("_options", []),
            "gold_label": entry.get("_gold_label", ""),
            "gold_text": entry.get("_gold_text", ""),
        }

    def evaluate_entry(self, output, entry):
        output_text = str(output)
        gold_label = entry.get("_gold_label", "")

        # 1. Extract explicit <answer>...</answer> tag content first.
        tag_content = extract_answer_tag(output_text)
        if tag_content:
            pred_label = _normalize_choice_label(tag_content)
            if pred_label and gold_label and pred_label == gold_label:
                return 1.0
            # Tag may contain full answer text instead of just a letter.
            pred_norm_tag = _normalize_text(tag_content)
            gold_text = _normalize_text(entry.get("_gold_text", ""))
            if pred_norm_tag and gold_text and gold_text in pred_norm_tag:
                return 1.0

        # 2. Fallback: first character of full output (model output only a letter).
        pred_label = _normalize_choice_label(output_text)
        if pred_label and gold_label and pred_label == gold_label:
            return 1.0

        # 3. Last resort: gold answer text substring in full output.
        pred_norm = _normalize_text(output_text)
        gold_text = _normalize_text(entry.get("_gold_text", ""))
        if pred_norm and gold_text and gold_text in pred_norm:
            return 1.0
        return 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "choices": entry.get("_options", []),
            "gold_label": entry.get("_gold_label", ""),
            "gold_text": entry.get("_gold_text", ""),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("GPQA uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("GPQA uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("GPQA uses evaluate_entry(output, entry).")
