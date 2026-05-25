from methods.all_history import AllHistory
from methods.DC_RS import DC_RS
from methods.ExpeL import ExpeL
from methods.ExpRecent import ExpRecent
from methods.ExpRAG import ExpRAG
from methods.janus import JanusMethod
from methods.janus_trigger_ablation import (
    JanusAlwaysTrigger,
    JanusPeriodicTrigger,
    JanusRandomTrigger,
)


_JANUS_HP_KEYS = {
    "janus_k",
    "janus_k_prime",
    "janus_l",
    "janus_beta",
    "janus_tau",
    "janus_seed",
    "janus_replay_limit",
}

# All baselines that can serve as the substitute base updater P for Janus.
_JANUS_BASE_REGISTRY = {
    "DC-RS": DC_RS,
    "DC_RS": DC_RS,
    "DCRS": DC_RS,
    "DynamicCheatsheetRetrievalSynthesis": DC_RS,
    "ExpeL": ExpeL,
}


def _split_janus_kwargs(kwargs):
    janus_kwargs = {
        "k": kwargs.pop("janus_k", 8),
        "k_prime": kwargs.pop("janus_k_prime", 3),
        "l_fresh": kwargs.pop("janus_l", 2),
        "beta": kwargs.pop("janus_beta", 0.9),
        "tau": kwargs.pop("janus_tau", 0.0),
        "seed": kwargs.pop("janus_seed", 0),
        "replay_limit": kwargs.pop("janus_replay_limit", None),
    }
    for key in list(kwargs.keys()):
        if key in _JANUS_HP_KEYS:
            kwargs.pop(key)
    return janus_kwargs, kwargs


def _resolve_janus_base(base_name):
    if base_name not in _JANUS_BASE_REGISTRY:
        valid = sorted(set(_JANUS_BASE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown janus_base '{base_name}'. Valid options: {valid}"
        )
    return _JANUS_BASE_REGISTRY[base_name]


def _build_janus(**kwargs):
    """Single factory: Janus wraps any registered base updater selected via
    the ``janus_base`` kwarg (defaults to DC-RS)."""
    kwargs = dict(kwargs)
    base_name = kwargs.pop("janus_base", "DC-RS")
    base_cls = _resolve_janus_base(base_name)
    janus_kwargs, base_kwargs = _split_janus_kwargs(kwargs)
    base = base_cls(**base_kwargs)
    return JanusMethod(base_method=base, **janus_kwargs)


def _build_janus_always(**kwargs):
    """Factory: Janus with always-trigger ablation."""
    kwargs = dict(kwargs)
    base_name = kwargs.pop("janus_base", "DC-RS")
    base_cls = _resolve_janus_base(base_name)
    janus_kwargs, base_kwargs = _split_janus_kwargs(kwargs)
    base = base_cls(**base_kwargs)
    return JanusAlwaysTrigger(base_method=base, **janus_kwargs)


def _build_janus_random(**kwargs):
    """Factory: Janus with random-trigger ablation."""
    kwargs = dict(kwargs)
    base_name = kwargs.pop("janus_base", "DC-RS")
    base_cls = _resolve_janus_base(base_name)
    trigger_rate = float(kwargs.pop("trigger_rate", 0.3))
    janus_kwargs, base_kwargs = _split_janus_kwargs(kwargs)
    base = base_cls(**base_kwargs)
    return JanusRandomTrigger(base_method=base, trigger_rate=trigger_rate, **janus_kwargs)


def _build_janus_periodic(**kwargs):
    """Factory: Janus with periodic-trigger ablation."""
    kwargs = dict(kwargs)
    base_name = kwargs.pop("janus_base", "DC-RS")
    base_cls = _resolve_janus_base(base_name)
    trigger_period = int(kwargs.pop("trigger_period", 3))
    janus_kwargs, base_kwargs = _split_janus_kwargs(kwargs)
    base = base_cls(**base_kwargs)
    return JanusPeriodicTrigger(base_method=base, trigger_period=trigger_period, **janus_kwargs)


METHOD_REGISTRY = {
    AllHistory.name: AllHistory,
    DC_RS.name: DC_RS,
    "DC_RS": DC_RS,
    "DC-RS": DC_RS,
    "DynamicCheatsheetRetrievalSynthesis": DC_RS,
    ExpeL.name: ExpeL,
    ExpRecent.name: ExpRecent,
    ExpRAG.name: ExpRAG,
    "ExpRAG": ExpRAG,
    "HistoryRAG": ExpRAG,
    # Single Janus entry; base updater P is selected via --janus-base.
    "Janus": _build_janus,
    # Trigger ablations (MMT Ablation Study 1).
    "JanusAlwaysTrigger": _build_janus_always,
    "JanusRandomTrigger": _build_janus_random,
    "JanusPeriodicTrigger": _build_janus_periodic,
}

__all__ = [
    "AllHistory",
    "DC_RS",
    "ExpRecent",
    "ExpRAG",
    "ExpeL",
    "JanusMethod",
    "JanusAlwaysTrigger",
    "JanusRandomTrigger",
    "JanusPeriodicTrigger",
    "METHOD_REGISTRY",
]
