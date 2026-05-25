import inspect

from methods import METHOD_REGISTRY
from tasks import TASK_REGISTRY


def build_method(method_name, **kwargs):
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")
    return METHOD_REGISTRY[method_name](**kwargs)


def _filter_kwargs_for_constructor(cls, kwargs):
    signature = inspect.signature(cls)
    params = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return dict(kwargs)
    accepted = {
        param.name
        for param in params
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def load_tasks(task_names, prompt_file=None, split=None, **kwargs):
    tasks = []
    base_kwargs = dict(kwargs)
    if prompt_file is not None:
        base_kwargs["prompt_file"] = prompt_file
    if split is not None:
        base_kwargs["split"] = split
    for name in task_names:
        if name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {name}")
        task_cls = TASK_REGISTRY[name]
        task_kwargs = _filter_kwargs_for_constructor(task_cls, base_kwargs)
        tasks.append(task_cls(**task_kwargs))
    return tasks
