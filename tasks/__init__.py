TASK_REGISTRY = {}

try:
    from tasks.task_math import MATHTask

    TASK_REGISTRY[MATHTask.name] = MATHTask
except ImportError:
    MATHTask = None

try:
    from tasks.task_math500 import MATH500Task

    TASK_REGISTRY[MATH500Task.name] = MATH500Task
except ImportError:
    MATH500Task = None

try:
    from tasks.task_humaneval import HumanEvalTask

    TASK_REGISTRY[HumanEvalTask.name] = HumanEvalTask
except ImportError:
    HumanEvalTask = None

try:
    from tasks.task_apibench import APIBenchHF

    TASK_REGISTRY[APIBenchHF.name] = APIBenchHF
except ImportError:
    APIBenchHF = None

try:
    from tasks.task_apibench import APIBenchTF

    TASK_REGISTRY[APIBenchTF.name] = APIBenchTF
except ImportError:
    APIBenchTF = None

try:
    from tasks.task_apibench import APIBenchTH

    TASK_REGISTRY[APIBenchTH.name] = APIBenchTH
except ImportError:
    APIBenchTH = None


try:
    from tasks.task_gpqa import GPQATask

    TASK_REGISTRY[GPQATask.name] = GPQATask
except ImportError:
    GPQATask = None

try:
    from tasks.task_mmlu_pro import MMLUProEngineeringTask, MMLUProPhysicsTask

    TASK_REGISTRY[MMLUProEngineeringTask.name] = MMLUProEngineeringTask
    TASK_REGISTRY[MMLUProPhysicsTask.name] = MMLUProPhysicsTask
    TASK_REGISTRY["MMLU-PRO-Engineering"] = MMLUProEngineeringTask
    TASK_REGISTRY["MMLU-PRO-Physics"] = MMLUProPhysicsTask
except ImportError:
    MMLUProEngineeringTask = None
    MMLUProPhysicsTask = None

__all__ = [
    "MATHTask",
    "MATH500Task",
    "GPQATask",
    "MMLUProEngineeringTask",
    "MMLUProPhysicsTask",
    "HumanEvalTask",
    "APIBenchHF",
    "APIBenchTF",
    "APIBenchTH",
    "TASK_REGISTRY",
]
