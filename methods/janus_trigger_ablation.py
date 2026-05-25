"""MMT trigger ablation variants for Janus.

Three drop-in subclasses of JanusMethod that swap out ONLY the trigger
decision while keeping the full comparison / restore / bookkeeping path
identical to the original:

  JanusAlwaysTrigger   – fires after every post-warmup step (100% rate)
  JanusRandomTrigger   – fires with fixed probability ``trigger_rate``
                         using a seeded, per-step RNG
  JanusPeriodicTrigger – fires every ``trigger_period`` post-warmup steps

All three still compute and log the actual MMT cosine similarity so that
metadata fields (``cosine_z_m_prev``) remain comparable across conditions.
"""

from __future__ import annotations

import random as _random

from methods.janus import JanusMethod
from methods.janus_core import cosine


class JanusAlwaysTrigger(JanusMethod):
    """Trigger at every post-warmup step regardless of trajectory."""

    name = "Janus-AlwaysTrigger"

    def _decide_trigger(self, z_t, prev_m, state) -> tuple:
        cos_val = float(cosine(z_t, prev_m)) if prev_m is not None else 0.0
        return True, cos_val


class JanusRandomTrigger(JanusMethod):
    """Trigger randomly with probability ``trigger_rate`` per step.

    Uses a deterministic per-step seed (``self.seed + step_count * 1009``)
    so results are reproducible while being uncorrelated across steps.
    Matches the overall trigger frequency of Janus when ``trigger_rate``
    is set to the observed MMT trigger rate on the same task.
    """

    name = "Janus-RandomTrigger"

    def __init__(self, *args, trigger_rate: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_rate = float(trigger_rate)

    def _decide_trigger(self, z_t, prev_m, state) -> tuple:
        cos_val = float(cosine(z_t, prev_m)) if prev_m is not None else 0.0
        rng = _random.Random(self.seed + state.step_count * 1009)
        triggered = rng.random() < self.trigger_rate
        return triggered, cos_val


class JanusPeriodicTrigger(JanusMethod):
    """Trigger every ``trigger_period`` post-warmup steps.

    Set ``trigger_period = round(post_warmup_steps / observed_trigger_count)``
    to match Janus's average trigger rate on a given task.
    """

    name = "Janus-PeriodicTrigger"

    def __init__(self, *args, trigger_period: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_period = max(1, int(trigger_period))

    def _decide_trigger(self, z_t, prev_m, state) -> tuple:
        cos_val = float(cosine(z_t, prev_m)) if prev_m is not None else 0.0
        # 0-indexed post-warmup step count
        post_idx = state.step_count - self.k - 1
        triggered = (post_idx % self.trigger_period == 0)
        return triggered, cos_val
