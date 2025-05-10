import functools
from abc import ABC
from itertools import chain
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from jl_transforms.run.runner import Runner

from jl_transforms.run.levels import Level


class Hook(ABC):
    def on_train_start(self, runner: "Runner"): ...

    def on_train_end(self, runner: "Runner"): ...

    def on_epoch_start(self, runner: "Runner"): ...

    def on_epoch_end(self, runner: "Runner"): ...

    def on_batch_start(self, runner: "Runner"): ...

    def on_batch_end(self, runner: "Runner"): ...


def run_hook(fn):
    name = fn.__name__

    if not name.startswith("_run_"):
        raise ValueError(
            f"Can only be used on methods starting with '_run_', got '{name}'"
        )

    hook_key = cast(Level, name[5:])

    @functools.wraps(fn)
    def wrapper(self: "Runner", *args, **kwargs):

        _call_hooks(self, hook_key, "start")
        ret_val = fn(self, *args, **kwargs)
        _call_hooks(self, hook_key, "end")

        return ret_val

    return wrapper


def _call_hooks(
    runner: "Runner", hook_key: Level, phase: Literal["start", "end"]
):
    method_name = f"on_{hook_key}_{phase}"
    for hook in chain(runner._internal_hooks, runner.hooks):
        method = getattr(hook, method_name, None)
        if callable(method):
            method(runner)
