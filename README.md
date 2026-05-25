# Janus: A Plug-in Controller for Sequentially Evolving LLM Memory

Janus is a plug-in **memory-deployment controller** for sequentially evolving LLM memory systems. In a sequential task stream, a base memory updater proposes a candidate memory after each task. Existing memory methods usually deploy every candidate update directly, but these local updates can make the final memory biased toward recent tasks, noisy rules, or over-specific insights.

Janus addresses this by deciding whether each candidate memory should be accepted or rejected. At step `t`, a base updater proposes a candidate memory $\hat{M} $. Janus either deploys it as the new memory $M_t$, or rolls back to the previous memory $M_{t-1}$. The goal is to maintain a final memory $M_T$ that is useful for future unseen tasks.

Janus has two main components:

1. **Memory Momentum Trigger (MMT)**: decides when old and new memory should be explicitly compared.
2. **Hybrid trigger-time evaluation set**: compares memory states on a compact set of coverage, boundary, and fresh tasks instead of replaying the full task history.

Janus is method-agnostic and currently supports wrapping `DC-RS` and `ExpeL`.

---

## Repository Structure

```text
.
|-- main.py                  # CLI entry point
|-- runner.py                # Sequential task loop
|-- config.py                # Default configuration
|-- timer.py                 # Timing utilities
|-- methods/                 # Memory methods and Janus controller
|   |-- janus.py
|   |-- janus_core.py
|   `-- ...
|-- tasks/                   # Dataset adapters and evaluators
|-- prompts/                 # Task and memory-update prompts
|-- utils/                   # LLM clients, logging, and I/O helpers
|-- data/                    # Local datasets when required
`-- outputs/                 # Experiment outputs
```

---

## Supported Methods

| Method | Description |
|---|---|
| `Memory-free` | Solves each task using only the base LLM, without persistent memory. |
| `ExpRAG` | Retrieves the top-k most similar past experiences by embedding similarity. |
| `DC-RS` | Dynamic Cheatsheet Retrieval-and-Synthesis; maintains a structured cheatsheet memory. |
| `ExpeL` | Extracts reusable insights from successful and failed trajectories through reflection. |
| `Janus-DC-RS` | Janus wrapping DC-RS as the base memory updater. |
| `Janus-ExpeL` | Janus wrapping ExpeL as the base memory updater. |

To run Janus, use:

```bash
--method Janus --janus-base DC-RS
```

or

```bash
--method Janus --janus-base ExpeL
```

The exported method name is `Janus-DC-RS` or `Janus-ExpeL`.

---

## Supported Tasks

The paper experiments use six datasets:

| Task name | Capability | Notes |
|---|---|---|
| `MATH` | Mathematical reasoning | Uses training examples for memory construction and MATH500 for testing. |
| `GPQA` | Graduate-level scientific reasoning | Uses GPQA Main for memory construction and GPQA Diamond for testing. |
| `MMLU_ENG` | Professional STEM reasoning | Engineering subset of MMLU-Pro. |
| `MMLU_PHY` | Professional STEM reasoning | Physics subset of MMLU-Pro. |
| `HumanEval` | Code generation | Uses part of the task set for memory construction and held-out problems for testing. |
| `APIBench-HF` | Tool/API use | HuggingFace subset of APIBench. |

Some datasets are loaded from HuggingFace, while others may require local files under `data/`. Check the corresponding adapter in `tasks/` for dataset-specific paths and preprocessing.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

Janus uses text embeddings for memory and task representations. Coverage-set clustering uses `kmeans-pytorch`. By default, clustering runs on CPU. To use GPU clustering when available, set:

```bash
export JANUS_KMEANS_DEVICE=cuda
```

---

## Quick Start

### Run DC-RS on MATH

```bash
python -u main.py \
  --method DC-RS \
  --tasks MATH \
  --generation-model Qwen/Qwen3-8B \
  --llm-backend vllm \
  --train-split train \
  --test-split test500 \
  --train-sample-size 500 \
  --sample-seed 0 \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --test-batch-size 128 \
  --enable-thinking \
  --output-dir outputs/MATH_DC_RS
```

### Run ExpeL on MMLU-Pro Engineering

```bash
python -u main.py \
  --method ExpeL \
  --tasks MMLU_ENG \
  --generation-model Qwen/Qwen3-8B \
  --llm-backend vllm \
  --train-split train \
  --test-split test \
  --sample-seed 0 \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --test-batch-size 128 \
  --enable-thinking \
  --output-dir outputs/MMLU_ENG_ExpeL
```

### Run Janus with DC-RS on GPQA

```bash
python -u main.py \
  --method Janus \
  --janus-base DC-RS \
  --tasks GPQA \
  --generation-model Qwen/Qwen3-8B \
  --llm-backend vllm \
  --gpqa-diamond-two-phase \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --test-batch-size 128 \
  --enable-thinking \
  --output-dir outputs/GPQA_Janus_DC_RS
```

### Run Janus with ExpeL on HumanEval

```bash
python -u main.py \
  --method Janus \
  --janus-base ExpeL \
  --tasks HumanEval \
  --generation-model Qwen/Qwen3-8B \
  --llm-backend vllm \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --test-batch-size 128 \
  --enable-thinking \
  --output-dir outputs/HumanEval_Janus_ExpeL
```

For a quick smoke test, add:

```bash
--run-first-k 5
```

---

## Janus Hyperparameters

The default Janus setting used in the paper is:

```text
K = 20
K' = 12
K_F = 5
tau = 0.0
```

where:

| Argument | Meaning |
|---|---|
| `--janus-k` | Total support-set size `K`. |
| `--janus-k-prime` | Coverage-set size `K' = |S_cov|`. |
| `--janus-l` | Fresh-set size `K_F = |F_t|`. |
| `--janus-beta` | Momentum EMA coefficient for MMT. |
| `--janus-tau` | MMT threshold. The trigger fires when `cos(z_t, m_{t-1}) < tau`. |
| `--janus-seed` | Random seed for Janus sampling. |
| `--janus-replay-limit` | Optional cap on the number of tasks evaluated per trigger. |

The hybrid trigger-time evaluation set is:

```text
E_t = S_cov_t ∪ S_bdry_t ∪ F_t
```

where:

- `S_cov_t` contains representative coverage tasks from the seen task stream.
- `S_bdry_t` contains memory-sensitive boundary tasks where old and new memory previously changed correctness.
- `F_t` contains fresh tasks sampled from recently encountered pending tasks.

The estimated trigger-time evaluation cost is:

```text
#Triggers × (K + K_F)
```

---

## How Janus Works

At each task step `t`:

1. The LLM solves the current task using the deployed memory `M_{t-1}`.
2. The base updater proposes a candidate memory `M_hat_t`.
3. Janus computes the memory update direction:

   ```text
   z_t = phi(M_hat_t) - phi(M_{t-1})
   ```

4. Janus maintains a memory momentum vector:

   ```text
   m_t = beta * m_{t-1} + (1 - beta) * z_t
   ```

5. The Memory Momentum Trigger fires when:

   ```text
   cos(z_t, m_{t-1}) < tau
   ```

6. If the trigger does not fire, Janus accepts the candidate memory directly.
7. If the trigger fires, Janus compares `M_{t-1}` and `M_hat_t` on the hybrid evaluation set.
8. Janus deploys the memory state with better evaluation performance.
9. Janus updates the coverage set, boundary set, and pending/fresh task pool.

This design avoids comparing old and new memory after every task, while still validating suspicious updates that may change the future behavior of the agent.

---

## Output Files

Each run writes outputs under the specified `--output-dir`.

Typical files include:

```text
<Task>_<Method>_memory.jsonl
<Task>_<Method>_memory_readable.json
<Task>_<Method>_stats.json
```

Depending on the experiment configuration, outputs may also be organized by split, for example:

```text
outputs/<run_name>/train/<Task>_<Method>_train_memory.jsonl
outputs/<run_name>/test/<Task>_<Method>_test_memory.jsonl
```

For each task, the memory file stores fields such as:

| Field | Description |
|---|---|
| `qid` | Task identifier. |
| `question` | Input task. |
| `model_output` | Raw model output. |
| `feedback` | Correctness or task-level feedback. |
| `score` | Evaluation score. |
| `memory` | Current deployed memory. |
| `memory_meta` | Method-specific metadata. |

For Janus runs, `memory_meta.janus` additionally records:

| Field | Description |
|---|---|
| `triggered` | Whether MMT fired at this step. |
| `chosen_memory` | Whether Janus chose old or new memory. |
| `cosine_z_m_prev` | Cosine similarity between current update and previous momentum. |
| `z_norm` | Norm of the current update direction. |
| `momentum_norm` | Norm of the memory momentum. |
| `support_old_acc` | Evaluation score of the old memory on the hybrid set. |
| `support_new_acc` | Evaluation score of the candidate memory on the hybrid set. |
| `num_eval_tasks` | Number of tasks used in trigger-time comparison. |
| `num_flip_tasks` | Number of memory-sensitive flip tasks. |
| `coverage_size` | Size of the coverage set. |
| `boundary_size` | Size of the boundary set. |
| `fresh_size` | Size of the fresh set. |
| `pending_size` | Number of pending tasks since the last trigger. |
| `seen_size` | Number of seen tasks. |
| `reason` | Decision reason, such as `trigger_old_better` or `trigger_new_better`. |

Janus may also export short previews:

```text
janus_old_memory_preview
janus_new_memory_preview
janus_deployed_memory_preview
```

These fields are useful for qualitative analysis.

---

## LLM Backends

The code supports multiple generation backends.

### Local models

Use HuggingFace or vLLM models, for example:

```bash
--generation-model Qwen/Qwen3-8B
--llm-backend vllm
```

Backend choices:

| Backend | Description |
|---|---|
| `transformers` | Use HuggingFace Transformers. |
| `vllm` | Force vLLM inference. |
| `auto` | Try vLLM first and fall back to Transformers if needed. |

### API models

For OpenAI-style models:

```bash
--generation-model openai/<model_name>
```

Required environment variable:

```bash
export OPENAI_API_KEY=<your_key>
```

For OpenRouter or other compatible providers, use the model identifier expected by your local client configuration.

---

## Common CLI Arguments

### General

| Argument | Description |
|---|---|
| `--method` | Memory method: `Memory-free`, `ExpRAG`, `DC-RS`, `ExpeL`, or `Janus`. |
| `--tasks` | Comma-separated task names. |
| `--generation-model` | LLM used for task solving. |
| `--curator-model` | Optional model for memory curation. Defaults to the generation model. |
| `--embedding-model` | Embedding model used for retrieval and Janus memory/task embeddings. |
| `--temperature` | Sampling temperature. |
| `--max-new-tokens` | Maximum number of generated tokens. |
| `--test-batch-size` | Batch size for evaluation. |
| `--enable-thinking` | Enable thinking mode for models such as Qwen3. |
| `--run-first-k` | Run only the first K examples for debugging. |
| `--output-dir` | Output directory. |
| `--timing` | Save per-stage timing statistics. |
| `--update-memory` / `--no-update-memory` | Enable or disable memory updates. |

### DC-RS

| Argument | Description |
|---|---|
| `--dc-rs-generator-prompt-path` | Override the DC-RS generator prompt. |
| `--dc-rs-curator-prompt-path` | Override the DC-RS curator prompt. |

Default DC-RS prompts are under:

```text
prompts/dc_rs/
```

### ExpeL

| Argument | Description |
|---|---|
| `--max-tries` | Maximum attempts per task. |
| `--batch-update-size` | Number of successful examples before a batch insight update. |
| `--insights-init` | Initial insight text. |
| `--reflection-model` | Optional reflection model. Defaults to the generation model. |
| `--max-num-rules` | Maximum number of maintained rules. |

### Janus

| Argument | Description |
|---|---|
| `--janus-base` | Base updater wrapped by Janus: `DC-RS` or `ExpeL`. |
| `--janus-k` | Total support-set size. |
| `--janus-k-prime` | Coverage-set size. |
| `--janus-l` | Fresh-set size. |
| `--janus-beta` | Momentum coefficient. |
| `--janus-tau` | MMT trigger threshold. |
| `--janus-seed` | Random seed for Janus sampling. |
| `--janus-replay-limit` | Optional cap on trigger-time replay size. |

When `--method Janus` is used, the CLI flags for the selected base updater are also honored.

---

## Reproducing the Paper Experiments

The main paper evaluates:

- Two LLMs: `Qwen3-8B` and `DeepSeek-V4-Flash`
- Six datasets: `MATH`, `GPQA`, `MMLU_ENG`, `MMLU_PHY`, `HumanEval`, `APIBench-HF`
- Two base updaters: `DC-RS` and `ExpeL`
- Two Janus variants: `Janus-DC-RS` and `Janus-ExpeL`

The default decoding setting is:

```text
max_new_tokens = 8192
temperature = 0.7
```

For Qwen3-8B, the paper uses thinking mode. For DeepSeek-V4-Flash, the paper uses non-thinking mode for cost efficiency.

A typical Janus run is:

```bash
python -u main.py \
  --method Janus \
  --janus-base DC-RS \
  --tasks GPQA \
  --generation-model Qwen/Qwen3-8B \
  --llm-backend vllm \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --enable-thinking \
  --janus-k 20 \
  --janus-k-prime 12 \
  --janus-l 5 \
  --janus-tau 0.0 \
  --output-dir outputs/GPQA_Janus_DC_RS
```

---

## Adding a New Base Updater to Janus

To wrap a new memory updater with Janus:

1. Implement the base memory updater following the repository method interface.
2. Add a `BaseJanusAdapter` subclass in `methods/janus.py`.
3. Implement adapter methods for:
   - reading the previous and candidate memory,
   - restoring old memory,
   - evaluating memory on replay tasks,
   - exporting Janus metadata.
4. Register the updater in the Janus base registry.

After registration, the new updater can be used with:

```bash
python -u main.py \
  --method Janus \
  --janus-base <NewUpdater> \
  --tasks <TaskName> \
  --generation-model <ModelName>
```

---

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{janus2026,
  title  = {The Past Is Prologue: A Plug-in Controller for Selective Updates in Sequentially Evolving LLM Memory},
  author = {Anonymous},
  year   = {2026},
  note   = {ACL submission}
}
```

---

## Notes

- Janus does not modify the underlying base updater. It controls whether each proposed memory should be deployed.
- Janus is designed for prompt-based sequential memory systems where the base LLM remains fixed.
- Full-history replay after every update can be expensive; Janus uses selective triggering and compact evaluation to reduce this cost.
- For stable timing comparisons, use the same backend policy across all methods in an experiment.
