import argparse
import json
import os
from pathlib import Path

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASKS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMING,
    DEFAULT_TOP_K,
    DEFAULT_UPDATE_MEMORY,
)
from batched_test import (
    dc_rs_base,
    run_batched_dc_rs,
    run_batched_test,
    supports_batched_test,
)
from registry import build_method, load_tasks
from runner import Runner
from timer import Timer
from utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Janus benchmark runner")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names (GSM8K, AIME2024, MATH, Omni-MATH, GPQA).",
    )
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--generation-model", type=str, default=DEFAULT_GENERATION_MODEL)
    parser.add_argument(
        "--curator-model",
        type=str,
        default=None,
        help="Optional curator model for DC-RS / ExpeL; defaults to generation model.",
    )
    parser.add_argument(
        "--dc-rs-generator-prompt-path",
        type=str,
        default=None,
        help="Optional generator prompt template path for DC-RS.",
    )
    parser.add_argument(
        "--dc-rs-curator-prompt-path",
        type=str,
        default=None,
        help="Optional curator prompt template path for DC-RS.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    # ------------------------------------------------------------- ExpeL args
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Maximum attempts per sample for ExpeL.",
    )
    parser.add_argument(
        "--batch-update-size",
        type=int,
        default=8,
        help="Batch size for ExpeL insight updates.",
    )
    parser.add_argument(
        "--insights-init",
        type=str,
        default="(empty)",
        help="Initial insight text for ExpeL.",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=None,
        help="Optional reflection model for ExpeL; defaults to generation model.",
    )
    parser.add_argument(
        "--max-num-rules",
        type=int,
        default=20,
        help="Maximum number of maintained insight rules for ExpeL.",
    )

    # ------------------------------------------------------------- Janus args
    parser.add_argument(
        "--janus-base",
        type=str,
        default="DC-RS",
        help=(
            "Base updater P for Janus (used only when --method Janus). "
            "Choices: DC-RS, ExpeL."
        ),
    )
    parser.add_argument(
        "--janus-k",
        type=int,
        default=20,
        help="Janus support set total size K.",
    )
    parser.add_argument(
        "--janus-k-prime",
        type=int,
        default=12,
        help="Janus coverage subset size K' (0 < K' <= K).",
    )
    parser.add_argument(
        "--janus-l",
        type=int,
        default=5,
        help="Janus fresh-set sample size L at trigger time.",
    )
    parser.add_argument(
        "--janus-beta",
        type=float,
        default=0.9,
        help="Janus momentum EMA coefficient beta.",
    )
    parser.add_argument(
        "--janus-tau",
        type=float,
        default=0.0,
        help="Janus MMT trigger threshold: fire when cos(z_t, m_{t-1}) < tau.",
    )
    parser.add_argument(
        "--janus-seed",
        type=int,
        default=0,
        help="Janus deterministic sampling seed.",
    )
    parser.add_argument(
        "--janus-replay-limit",
        type=int,
        default=None,
        help="Optional cap on number of support/fresh tasks evaluated at each trigger.",
    )


    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override the dataset split for single-phase runs (e.g., test, train).",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help=(
            "Train-stream split for two-phase runs. When set together with "
            "--test-split, memory is built on this split first."
        ),
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help=(
            "Test split for two-phase runs. When set together with "
            "--train-split, the final memory is evaluated here without updates."
        ),
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=1000,
        help="Cap on number of train samples used for memory building.",
    )
    parser.add_argument(
        "--sample-seed",
        "--seed",
        dest="sample_seed",
        type=int,
        default=0,
        help=(
            "Seed for deterministic dataset sampling. This controls the "
            "train split shuffle and task-level sampled splits such as "
            "MMLU-Pro train/test slices."
        ),
    )
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for shuffling train-stream order only "
            "(two-phase train with --shuffle via Runner). Defaults to "
            "--sample-seed. Use a fixed --sample-seed and vary this flag "
            "for stream-order / recency-bias studies without changing the "
            "train/test partition."
        ),
    )

    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["auto", "vllm", "transformers"],
        default="vllm",
        help=(
            "Optional local backend override for utils.llm_client "
            "(sets LLM_BACKEND for this run)."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode in chat template when supported.",
    )
    parser.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode in chat template when supported.",
    )
    parser.add_argument(
        "--update-memory",
        action="store_true",
        default=DEFAULT_UPDATE_MEMORY,
    )
    parser.add_argument(
        "--no-update-memory",
        dest="update_memory",
        action="store_false",
    )
    parser.add_argument("--timing", action="store_true", default=DEFAULT_TIMING)
    parser.add_argument(
        "--run-first-k",
        type=int,
        default=None,
        help="Run only the first K samples for each task.",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size used for the two-phase test pass (DC-RS / "
            "Janus-DC_RS only). Submits this many prompts per vLLM call "
            "to maximize throughput. Ignored for other methods."
        ),
    )
    parser.add_argument(
        "--test-no-batch",
        action="store_true",
        default=False,
        help=(
            "Disable the batched DC-RS test fast path and fall back to "
            "the legacy per-sample Runner for the test phase."
        ),
    )
    parser.add_argument(
        "--test-use-curator",
        action="store_true",
        default=False,
        help=(
            "Run the per-question curator pass at test time (original "
            "DC-RS behaviour). Default: disabled -- use the frozen "
            "cheatsheet from the train phase as-is."
        ),
    )
    parser.add_argument(
        "--gpqa-diamond-two-phase",
        action="store_true",
        default=False,
        help=(
            "Shortcut for GPQA: train on gpqa_main\\\\gpqa_diamond and "
            "test on gpqa_diamond."
        ),
    )
    # Trigger ablation knobs (MMT Ablation Study 1).
    parser.add_argument(
        "--ablation-trigger-rate",
        dest="ablation_trigger_rate",
        type=float,
        default=0.3,
        help=(
            "JanusRandomTrigger: probability of triggering at each "
            "post-warmup step. Set to the observed MMT trigger rate on "
            "the same task to match Janus's overall trigger frequency. "
            "Default: 0.3."
        ),
    )
    parser.add_argument(
        "--ablation-trigger-period",
        dest="ablation_trigger_period",
        type=int,
        default=3,
        help=(
            "JanusPeriodicTrigger: trigger every N post-warmup steps. "
            "Set N = round(post_warmup_steps / observed_trigger_count) "
            "to match Janus's average trigger rate. Default: 3."
        ),
    )
    return parser.parse_args()


def _dc_rs_kwargs_from_args(args, method_kwargs):
    if args.curator_model is not None:
        method_kwargs["curator_model_name"] = args.curator_model
    if args.dc_rs_generator_prompt_path is not None:
        method_kwargs["generator_prompt_path"] = args.dc_rs_generator_prompt_path
    if args.dc_rs_curator_prompt_path is not None:
        method_kwargs["curator_prompt_path"] = args.dc_rs_curator_prompt_path


def _expel_kwargs_from_args(args, method_kwargs):
    method_kwargs["max_tries"] = args.max_tries
    method_kwargs["batch_update_size"] = args.batch_update_size
    method_kwargs["insights_init"] = args.insights_init
    method_kwargs["max_num_rules"] = args.max_num_rules
    if args.reflection_model is not None:
        method_kwargs["reflection_model_name"] = args.reflection_model
    if args.curator_model is not None:
        method_kwargs["curator_model_name"] = args.curator_model


def main():
    setup_logging()
    args = parse_args()
    if args.llm_backend is not None:
        os.environ["LLM_BACKEND"] = args.llm_backend
    if args.enable_thinking is not None:
        os.environ["ENABLE_THINKING"] = "true" if args.enable_thinking else "false"
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    if args.gpqa_diamond_two_phase:
        if len(task_names) != 1 or task_names[0] != "GPQA":
            raise ValueError("--gpqa-diamond-two-phase requires --tasks GPQA.")
        if args.split is not None:
            raise ValueError(
                "--gpqa-diamond-two-phase cannot be combined with --split."
            )
        if args.train_split is None:
            args.train_split = "main_wo_diamond"
        if args.test_split is None:
            args.test_split = "diamond"

    method_kwargs = {
        "top_k": args.top_k,
        "embedding_model_name": args.embedding_model,
        "generation_model_name": args.generation_model,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.method in {"DC-RS", "DC_RS", "DynamicCheatsheetRetrievalSynthesis"}:
        _dc_rs_kwargs_from_args(args, method_kwargs)

    if args.method == "ExpeL":
        _expel_kwargs_from_args(args, method_kwargs)

    _JANUS_FAMILY = {
        "Janus",
        "JanusAlwaysTrigger", "JanusRandomTrigger", "JanusPeriodicTrigger",
    }

    if args.method in _JANUS_FAMILY:
        method_kwargs["janus_base"] = args.janus_base
        method_kwargs["janus_k"] = args.janus_k
        method_kwargs["janus_k_prime"] = args.janus_k_prime
        method_kwargs["janus_l"] = args.janus_l
        method_kwargs["janus_beta"] = args.janus_beta
        method_kwargs["janus_tau"] = args.janus_tau
        method_kwargs["janus_seed"] = args.janus_seed
        if args.janus_replay_limit is not None:
            method_kwargs["janus_replay_limit"] = args.janus_replay_limit


        # Ablation-specific knobs (silently ignored by Janus factories).
        method_kwargs["trigger_rate"] = args.ablation_trigger_rate
        method_kwargs["trigger_period"] = args.ablation_trigger_period

        base_choice = args.janus_base
        if base_choice in {"DC-RS", "DC_RS", "DCRS", "DynamicCheatsheetRetrievalSynthesis"}:
            _dc_rs_kwargs_from_args(args, method_kwargs)
        elif base_choice == "ExpeL":
            _expel_kwargs_from_args(args, method_kwargs)
        else:
            raise ValueError(
                f"Unknown --janus-base value: {base_choice}. Expected DC-RS or ExpeL."
            )

    method = build_method(args.method, **method_kwargs)
    timer = Timer(enabled=args.timing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    two_phase = bool(args.train_split) and bool(args.test_split)
    all_results = []

    if two_phase:
        train_tasks = load_tasks(
            task_names,
            prompt_file=args.prompt_file,
            split=args.train_split,
            sample_seed=args.sample_seed,
        )
        test_tasks = load_tasks(
            task_names,
            prompt_file=args.prompt_file,
            split=args.test_split,
            sample_seed=args.sample_seed,
        )

        train_dir = output_dir / "train"
        test_dir = output_dir / "test"

        train_shuffle_seed = (
            args.train_shuffle_seed
            if args.train_shuffle_seed is not None
            else args.sample_seed
        )
        train_runner = Runner(method, train_tasks, timer, output_dir=train_dir)
        train_results = train_runner.run(
            update_memory=True,
            run_first_k=args.run_first_k,
            sample_size=args.train_sample_size,
            sample_seed=train_shuffle_seed,
            shuffle_sampling=True,
            reset_task_state=True,
            phase_label="train",
            log_predictions=False,
        )

        want_curator = bool(args.test_use_curator) and (dc_rs_base(method) is not None)
        can_batch = supports_batched_test(method) or (dc_rs_base(method) is not None)
        use_batched_test = (not args.test_no_batch) and can_batch

        if use_batched_test:
            test_dir.mkdir(parents=True, exist_ok=True)
            if want_curator:
                print(
                    f"[main] using batched DC-RS test with curator, "
                    f"batch_size={args.test_batch_size}"
                )
                test_results = run_batched_dc_rs(
                    method=method,
                    tasks=test_tasks,
                    output_dir=test_dir,
                    batch_size=args.test_batch_size,
                    run_first_k=args.run_first_k,
                    use_curator=True,
                    phase_label="test",
                )
            else:
                print(
                    f"[main] using generic batched test (frozen memory), "
                    f"method={method.name}, batch_size={args.test_batch_size}"
                )
                test_results = run_batched_test(
                    method=method,
                    tasks=test_tasks,
                    output_dir=test_dir,
                    batch_size=args.test_batch_size,
                    run_first_k=args.run_first_k,
                    phase_label="test",
                )
        else:
            print(
                f"[main] using legacy per-sample Runner for test (method={method.name})"
            )
            test_runner = Runner(method, test_tasks, timer, output_dir=test_dir)
            test_results = test_runner.run(
                update_memory=False,
                run_first_k=args.run_first_k,
                reset_task_state=False,
                phase_label="test",
                log_predictions=True,
            )

        all_results = list(train_results) + list(test_results)
        results_payload = {"train": train_results, "test": test_results}
    else:
        tasks = load_tasks(
            task_names,
            prompt_file=args.prompt_file,
            split=args.split,
            sample_seed=args.sample_seed,
        )
        runner = Runner(method, tasks, timer, output_dir=output_dir)
        results = runner.run(
            update_memory=args.update_memory, run_first_k=args.run_first_k
        )
        all_results = list(results)
        results_payload = results

    print(json.dumps(results_payload, ensure_ascii=False, indent=2))

    stage_avg = timer.summary() if args.timing else {}
    stage_avg_by_task = {}
    per_sample_avg_seconds = {}
    sample_total_avg_seconds = {}
    for key, avg_sec in stage_avg.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, task_name, stage_name = parts
        stage_avg_by_task.setdefault(task_name, {})[stage_name] = avg_sec
        if stage_name in {"retrieve", "generate", "update"}:
            per_sample_avg_seconds[task_name] = (
                per_sample_avg_seconds.get(task_name, 0.0) + avg_sec
            )
        if stage_name == "sample_total":
            sample_total_avg_seconds[task_name] = float(avg_sec)

    saved_stats_paths = []
    for task_result in all_results:
        task_name = task_result["task"]
        phase = task_result.get("phase")
        stats = {
            "task": task_name,
            "method": task_result["method"],
            "phase": phase,
            "total": task_result["total"],
            "success": task_result["success"],
            "accuracy": task_result["accuracy"],
        }
        if args.timing:
            stats["stage_avg_seconds"] = stage_avg_by_task.get(task_name, {})
            stats["per_sample_avg_seconds"] = per_sample_avg_seconds.get(task_name, 0.0)
            stats["sample_total_avg_seconds"] = sample_total_avg_seconds.get(task_name, 0.0)

        if phase:
            stats_dir = output_dir / phase
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats_path = stats_dir / f"{task_name}_{method.name}_{phase}_stats.json"
        else:
            stats_path = output_dir / f"{task_name}_{method.name}_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        saved_stats_paths.append(str(stats_path))

    print(json.dumps({"saved_stats_files": saved_stats_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
