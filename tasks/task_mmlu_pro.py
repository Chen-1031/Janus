import hashlib
import random
import re

from datasets import load_dataset

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import extract_answer_tag


_LABELS = list("ABCDEFGHIJ")


def _normalize_text(text):
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_choice_label(text, valid_labels=None):
    valid = set(valid_labels or _LABELS)
    text = "" if text is None else str(text).strip()
    if not text:
        return ""

    upper = text.upper()
    # Prefer explicit final-answer phrasing when present.
    patterns = [
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-J])\)?",
        r"\\BOXED\{\s*([A-J])\s*\}",
        r"<FINAL_ANSWER>\s*([A-J])\s*</FINAL_ANSWER>",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, upper)
        for label in reversed(matches):
            if label in valid:
                return label

    # If the model obeyed the instruction and emitted just a label.
    direct = re.fullmatch(r"\(?([A-J])\)?\.?", upper)
    if direct and direct.group(1) in valid:
        return direct.group(1)

    # Last resort: use the last standalone option label in the output.
    matches = re.findall(r"\b([A-J])\b", upper)
    for label in reversed(matches):
        if label in valid:
            return label
    return ""


class MMLUProTask(BaseTask):
    prompt_key = "mmlu_pro"

    def __init__(
        self,
        prompt_file=None,
        hf_path="TIGER-Lab/MMLU-Pro",
        category=None,
        split="test",
        hf_split="test",
        sample_size=250,
        sample_seed=0,
    ):
        self.prompt_file = prompt_file
        self.hf_path = hf_path
        self.category = category
        self.split = str(split or "test").strip().lower()
        self.hf_split = hf_split
        self.sample_size = int(sample_size)
        self.sample_seed = int(sample_seed)
        self._entries = None

    def _phase_offset(self):
        if self.split in {"train", "trainset", "memory", "support"}:
            return 0
        if self.split in {"test", "testset", "eval", "validation", "val"}:
            return self.sample_size
        if self.split in {"full", "all", "raw"}:
            return None
        raise ValueError(
            "Unsupported MMLU-Pro split. Use train, test, or full. "
            "The train/test aliases are deterministic disjoint 250-sample "
            "slices from the upstream test split."
        )

    def _question(self, item):
        return item.get("question", "") or item.get("Question", "")

    def _category(self, item):
        return (
            item.get("category", "")
            or item.get("Category", "")
            or item.get("subject", "")
            or item.get("Subject", "")
        )

    def _raw_options(self, item):
        options = item.get("options")
        if options is None:
            options = item.get("Options")
        if isinstance(options, dict):
            return [options[k] for k in sorted(options.keys())]
        if isinstance(options, (list, tuple)):
            return list(options)

        collected = []
        for label in _LABELS:
            for key in (
                label,
                label.lower(),
                f"option_{label.lower()}",
                f"option_{label}",
            ):
                if key in item and str(item.get(key, "")).strip():
                    collected.append(item[key])
                    break
        return collected

    def _options(self, item):
        options = [str(opt).strip() for opt in self._raw_options(item)]
        return [(label, text) for label, text in zip(_LABELS, options) if text]

    def _gold_label(self, item):
        options = self._options(item)
        valid = [label for label, _ in options]

        for key in ("answer", "Answer", "gold", "label", "target"):
            if key not in item:
                continue
            label = _normalize_choice_label(item.get(key), valid)
            if label:
                return label

        for key in ("answer_index", "answer_idx", "label_index", "target_index"):
            if key not in item:
                continue
            try:
                idx = int(item.get(key))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(valid):
                return valid[idx]

        answer_text = _normalize_text(
            item.get("answer")
            or item.get("Answer")
            or item.get("correct_answer")
            or item.get("Correct Answer")
        )
        for label, option_text in options:
            if answer_text and answer_text == _normalize_text(option_text):
                return label
        return ""

    def _qid(self, item, idx):
        for key in ("question_id", "id", "ID", "qid"):
            if key in item:
                return str(item[key])
        category = _normalize_text(self._category(item)).replace(" ", "_")
        return f"{category}-{idx}"

    def _stable_shuffle(self, rows):
        category_key = _normalize_text(self.category)
        digest = hashlib.sha256(
            f"{self.hf_path}:{category_key}:{self.sample_seed}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        rows = list(rows)
        rng.shuffle(rows)
        return rows

    def _load(self):
        if self._entries is not None:
            return

        dataset = load_dataset(self.hf_path, split=self.hf_split)
        rows = [dict(item) for item in dataset]
        if self.category:
            wanted = _normalize_text(self.category)
            rows = [
                item
                for item in rows
                if _normalize_text(self._category(item)) == wanted
            ]

        rows = self._stable_shuffle(rows)
        offset = self._phase_offset()
        if offset is not None:
            need = offset + self.sample_size
            if len(rows) < need:
                raise RuntimeError(
                    f"MMLU-Pro category={self.category!r} has only {len(rows)} "
                    f"rows, but split={self.split!r} requires at least {need}."
                )
            rows = rows[offset : offset + self.sample_size]

        entries = []
        for idx, item in enumerate(rows):
            prepared = dict(item)
            prepared["_qid"] = self._qid(prepared, idx)
            prepared["_options"] = self._options(prepared)
            prepared["_gold_label"] = self._gold_label(prepared)
            prepared["_category"] = self._category(prepared)
            if not prepared["_options"] or not prepared["_gold_label"]:
                raise RuntimeError(
                    "MMLU-Pro sample is missing options or a gold answer: "
                    f"qid={prepared['_qid']}"
                )
            entries.append(prepared)
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
        valid_labels = "/".join([k for k, _ in entry.get("_options", [])])
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
            choices=choices_text,
            valid_labels=valid_labels,
        )

    def build_inputs(self, entry):
        gold_label = entry.get("_gold_label", "")
        gold_text = dict(entry.get("_options", [])).get(gold_label, "")
        return {
            "qid": entry.get("_qid", ""),
            "category": entry.get("_category", ""),
            "question": self._question(entry),
            "choices": entry.get("_options", []),
            "gold_label": gold_label,
            "gold_text": gold_text,
        }

    def evaluate_entry(self, output, entry):
        valid = [label for label, _ in entry.get("_options", [])]
        gold_label = entry.get("_gold_label", "")

        # 1. Prefer an explicit <answer> tag — most reliable signal.
        tag_content = extract_answer_tag(str(output))
        if tag_content:
            pred_label = _normalize_choice_label(tag_content, valid)
            if pred_label and gold_label and pred_label == gold_label:
                return 1.0

        # 2. Fall back to scanning the full output for the label.
        pred_label = _normalize_choice_label(output, valid)
        if pred_label and gold_label and pred_label == gold_label:
            return 1.0

        pred_norm = _normalize_text(output)
        gold_text = _normalize_text(dict(entry.get("_options", [])).get(gold_label, ""))
        if pred_norm and gold_text and gold_text in pred_norm:
            return 1.0
        return 0.0

    def build_memory_record(self, entry, output, feedback, score):
        inputs = self.build_inputs(entry)
        return {
            "qid": inputs["qid"],
            "category": inputs["category"],
            "question": inputs["question"],
            "choices": inputs["choices"],
            "gold_label": inputs["gold_label"],
            "gold_text": inputs["gold_text"],
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("MMLU-Pro uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("MMLU-Pro uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("MMLU-Pro uses evaluate_entry(output, entry).")


class MMLUProEngineeringTask(MMLUProTask):
    name = "MMLU_ENG"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("category", "engineering")
        super().__init__(*args, **kwargs)


class MMLUProPhysicsTask(MMLUProTask):
    name = "MMLU_PHY"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("category", "physics")
        super().__init__(*args, **kwargs)
