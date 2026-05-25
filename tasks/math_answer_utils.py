import re


def extract_last_boxed(text):
    if text is None:
        return ""
    raw = str(text)
    token = r"\boxed{"
    start_positions = []
    offset = 0
    while True:
        idx = raw.find(token, offset)
        if idx < 0:
            break
        start_positions.append(idx)
        offset = idx + 1
    if not start_positions:
        return ""

    start = start_positions[-1] + len(token)
    depth = 1
    i = start
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i].strip()
        i += 1
    return ""


def _strip_outer_delimiters(text):
    s = text.strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        if (s.startswith("$") and s.endswith("$")) or (
            s.startswith("\\(") and s.endswith("\\)")
        ) or (s.startswith("\\[") and s.endswith("\\]")):
            if s.startswith("$"):
                s = s[1:-1].strip()
            else:
                s = s[2:-2].strip()
            changed = True
    return s


def _strip_balanced_outer_braces(text):
    s = text.strip()
    while s.startswith("{") and s.endswith("}"):
        depth = 0
        balanced = True
        for idx, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and idx != len(s) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        s = s[1:-1].strip()
    return s


def normalize_math_expr(text):
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    s = _strip_outer_delimiters(s)
    s = _strip_balanced_outer_braces(s)
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace(r"\,", "")
    s = s.replace(r"\!", "")
    s = s.strip()
    s = re.sub(r"[.;:,]+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def extract_answer_tag(text):
    """Extract the answer from the last recognized answer tag.

    Supported formats (in priority order):
      - <answer>...</answer>              (DC_RS / general prompt format)
      - <final_answer>...</final_answer>  (ExpeL format)
    """
    if text is None:
        return ""
    raw = str(text)
    for tag in ("answer", "final_answer"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(raw)
        if matches:
            return matches[-1].strip()
    return ""


def normalized_last_boxed(text):
    boxed = extract_last_boxed(text)
    if boxed:
        return normalize_math_expr(boxed)
    # Fallback: model may have used <answer> tags instead of \boxed{}.
    tag_content = extract_answer_tag(text)
    if tag_content:
        return normalize_math_expr(tag_content)
    return ""


def normalized_gold_answer(answer_text, fallback_solution_text=None):
    primary = str(answer_text or "").strip()
    if primary:
        if r"\boxed{" in primary:
            return normalized_last_boxed(primary)
        return normalize_math_expr(primary)
    if fallback_solution_text:
        return normalized_last_boxed(fallback_solution_text)
    return ""
