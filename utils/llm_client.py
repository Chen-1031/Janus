from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Keep transformers on PyTorch path in mixed environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False




try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


_CACHE: Dict[str, Tuple[object, object]] = {}
_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# Hardcoded sampling penalties (not exposed as experiment hyperparameters).
# vLLM supports both; HF transformers only supports repetition_penalty natively.
_PRESENCE_PENALTY = 1.0
_REPETITION_PENALTY = 1.0
_ENABLE_THINKING_DEFAULT = False
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "OpenRouter")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
OPENROUTER_TOP_P = 0.95
OPENROUTER_TOP_K = 20
OPENROUTER_PRESENCE_PENALTY = 1.0
OPENROUTER_FREQUENCY_PENALTY = 1.0
OPENROUTER_MIN_P = 0.0
OPENROUTER_REPETITION_PENALTY = 1.0
OPENROUTER_MAX_WORKERS = 32


def _resolve_enable_thinking_default() -> bool:
    raw = os.getenv("ENABLE_THINKING")
    if raw is None:
        return _ENABLE_THINKING_DEFAULT
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return _ENABLE_THINKING_DEFAULT


def _resolve_provider(model_name: str) -> str:
    model_key = model_name.strip().lower()
    if model_key.startswith("openrouter/") or model_key == "openrouter":
        # print("openrouter")
        return "openrouter"
    if model_key.startswith("openai/") or model_key in {"openai", "chatgpt"}:
        return "openai"
    return "local"

def _resolve_openrouter_model(model_name: str) -> str:
    model_key = model_name.strip()
    if model_key.lower().startswith("openrouter/"):
        # Keep everything after the first "openrouter/" so provider/model
        # forms like "openrouter/openai/gpt-5.2" remain intact.
        remainder = model_key.split("/", 1)[1].strip()
        if not remainder:
            raise RuntimeError(
                "OpenRouter model name must include a provider/model suffix, "
                "e.g. openrouter/openai/gpt-5.2"
            )
        return remainder
    if model_key.lower() == "openrouter":
        default = OPENROUTER_MODEL
        if not default:
            raise RuntimeError(
                "OPENROUTER_MODEL is not set. Either pass "
                "--generation-model openrouter/<provider>/<model> "
                "or export OPENROUTER_MODEL=<provider>/<model>."
            )
        return default
    return model_key


def _generate_openrouter(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    enable_thinking: bool | None = None,
) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("openai SDK not installed. Please install: pip install openai")
    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in your shell before "
            "using OpenRouter models."
        )

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    or_model = _resolve_openrouter_model(model_name)

    extra_headers: Dict[str, str] = {
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-OpenRouter-Title": OPENROUTER_SITE_NAME,
    }

    # Align with local backends: caller-level enable_thinking takes precedence,
    # then fall back to the global ENABLE_THINKING default resolver.
    thinking_flag = (
        _resolve_enable_thinking_default()
        if enable_thinking is None
        else bool(enable_thinking)
    )

    # extra_body: Dict[str, object] = {
    #     "reasoning": {"enabled": thinking_flag, "effort": "low"},
    #     "top_k": OPENROUTER_TOP_K,
    #     "min_p": OPENROUTER_MIN_P,
    #     "repetition_penalty": OPENROUTER_REPETITION_PENALTY,
    # }

    # sampling_kwargs: Dict[str, float] = {
    #     "top_p": OPENROUTER_TOP_P,
    #     "presence_penalty": OPENROUTER_PRESENCE_PENALTY,
    #     "frequency_penalty": OPENROUTER_FREQUENCY_PENALTY,
    # }

    
    extra_body: Dict[str, object] = {
        "reasoning": {"enabled": thinking_flag, "effort": "none"},
    }

    sampling_kwargs: Dict[str, float] = {
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
    }

    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": or_model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": float(temperature),
            }
            kwargs.update(sampling_kwargs)
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            return (content or "").strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"OpenRouter error (try {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"OpenRouter request failed after retries: {exc}"
                ) from exc

    return ""

def _resolve_local_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "transformers").strip().lower()
    if backend not in {"auto", "vllm", "transformers"}:
        raise RuntimeError(
            "Invalid LLM_BACKEND. Expected one of: auto, vllm, transformers"
        )
    return backend


def _resolve_openai_model(model_name: str) -> str:
    model_key = model_name.strip()
    if model_key.lower().startswith("openai/"):
        return model_key.split("/", 1)[1]
    if model_key.lower() in {"openai", "chatgpt"}:
        return os.getenv("OPENAI_MODEL", "gpt-5.2")
    return model_key


def _generate_openai(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("openai not installed. Please install: pip install openai")
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell before "
            "using OpenAI models."
        )

    client = OpenAI(api_key=api_key)
    openai_model = _resolve_openai_model(model_name)

    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=openai_model,
                input=prompt,
                max_output_tokens=max_new_tokens,
                reasoning={"effort": "none"},
            )
            return (resp.output_text or "").strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"OpenAI error (try {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(f"OpenAI request failed after retries: {exc}") from exc

    return ""


def _load_local_model_with_vllm(model_name: str):
    key = f"{model_name}|vllm"
    tok, llm = _CACHE.get(key, (None, None))
    if tok is not None and llm is not None:
        return tok, llm

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "2"))
    gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.6"))
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": max(1, tp),
        "gpu_memory_utilization": min(0.99, max(0.01, gpu_mem)),
    }
    # By default we do not pass max_model_len, so vLLM can derive it from model config.
    user_max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")
    if user_max_model_len:
        llm_kwargs["max_model_len"] = int(user_max_model_len)

    llm = LLM(**llm_kwargs)
    _CACHE[key] = (tok, llm)
    return tok, llm


def _generate_local_with_vllm(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    tok, llm = _load_local_model_with_vllm(model_name)

    if hasattr(tok, "apply_chat_template"):
        thinking_flag = (
            _resolve_enable_thinking_default()
            if enable_thinking is None
            else bool(enable_thinking)
        )
        try:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        chat_text = prompt

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=float(temperature),
        n=1,
        presence_penalty=(
            _PRESENCE_PENALTY
            if presence_penalty is None
            else float(presence_penalty)
        ),
        repetition_penalty=(
            _REPETITION_PENALTY
            if repetition_penalty is None
            else float(repetition_penalty)
        ),
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )
    outputs = llm.generate([chat_text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def _load_local_model_with_transformers(model_name: str):
    key = f"{model_name}|hf"
    tok, mdl = _CACHE.get(key, (None, None))
    if tok is not None and mdl is not None:
        return tok, mdl

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    mdl.eval()
    _CACHE[key] = (tok, mdl)
    return tok, mdl


def _generate_local_with_transformers(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    tok, mdl = _load_local_model_with_transformers(model_name)

    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if hasattr(tok, "apply_chat_template"):
        thinking_flag = (
            _resolve_enable_thinking_default()
            if enable_thinking is None
            else bool(enable_thinking)
        )
        try:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        chat_text = prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    inputs = tok(chat_text, return_tensors="pt").to(device)
    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=float(temperature),
        repetition_penalty=(
            _REPETITION_PENALTY
            if repetition_penalty is None
            else float(repetition_penalty)
        ),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def _chat_text(tok, prompt: str, enable_thinking: bool | None = None) -> str:
    if hasattr(tok, "apply_chat_template"):
        thinking_flag = (
            _resolve_enable_thinking_default()
            if enable_thinking is None
            else bool(enable_thinking)
        )
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def _generate_batch_vllm(
    model_name: str,
    prompts,
    temperature: float,
    max_new_tokens: int,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
):
    tok, llm = _load_local_model_with_vllm(model_name)
    chat_texts = [_chat_text(tok, p, enable_thinking=enable_thinking) for p in prompts]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=float(temperature),
        n=1,
        presence_penalty=(
            _PRESENCE_PENALTY
            if presence_penalty is None
            else float(presence_penalty)
        ),
        repetition_penalty=(
            _REPETITION_PENALTY
            if repetition_penalty is None
            else float(repetition_penalty)
        ),
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
        # top_p=0.95, ##test with qwen tech report
        # top_k=20  ##test with qwen tech report
    )
    outputs = llm.generate(chat_texts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


def generate_batch(
    model_name: str,
    prompts,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
):
    """Batched version of :func:`generate`.

    For vLLM this submits the whole list in a single call so vLLM can
    schedule them concurrently (continuous batching). For other backends
    it simply falls back to looping :func:`generate`.
    """
    prompts = list(prompts)
    if not prompts:
        return []

    provider = _resolve_provider(model_name)
    if provider == "openai":
        return [
            _generate_openai(model_name, p, max_new_tokens) for p in prompts
        ]
    if provider == "openrouter":
        results: List[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=OPENROUTER_MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    _generate_openrouter,
                    model_name, p, temperature, max_new_tokens,
                    enable_thinking=enable_thinking,
                ): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results

    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "Local generation requires transformers and torch. "
            "Please install: pip install transformers torch"
        )

    backend = _resolve_local_backend()
    if backend == "transformers":
        return [
            _generate_local_with_transformers(
                model_name,
                p,
                temperature,
                max_new_tokens,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
            )
            for p in prompts
        ]

    # vllm / auto
    if _HAS_VLLM:
        try:
            return _generate_batch_vllm(
                model_name,
                prompts,
                temperature,
                max_new_tokens,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            print(f"vLLM batch failed, fallback to transformers loop: {exc}")

    return [
        _generate_local_with_transformers(
            model_name,
            p,
            temperature,
            max_new_tokens,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )
        for p in prompts
    ]


def generate(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    provider = _resolve_provider(model_name)
    if provider == "openai":
        return _generate_openai(model_name, prompt, max_new_tokens)
    if provider == "openrouter":
        return _generate_openrouter(
            model_name,
            prompt,
            temperature,
            max_new_tokens,
            enable_thinking=enable_thinking,
        )

    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "Local generation requires transformers and torch. "
            "Please install: pip install transformers torch"
        )

    backend = _resolve_local_backend()

    if backend == "vllm":
        if not _HAS_VLLM:
            raise RuntimeError(
                "LLM_BACKEND=vllm is set, but vllm is not available. "
                "Please install: pip install vllm"
            )
        return _generate_local_with_vllm(
            model_name,
            prompt,
            temperature,
            max_new_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )

    if backend == "transformers":
        return _generate_local_with_transformers(
            model_name,
            prompt,
            temperature,
            max_new_tokens,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )

    # auto: prefer vLLM, fallback to transformers when vLLM fails at runtime.
    if _HAS_VLLM:
        try:
            return _generate_local_with_vllm(
                model_name,
                prompt,
                temperature,
                max_new_tokens,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            print(f"vLLM failed, fallback to transformers: {exc}")

    return _generate_local_with_transformers(
        model_name,
        prompt,
        temperature,
        max_new_tokens,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
    )
