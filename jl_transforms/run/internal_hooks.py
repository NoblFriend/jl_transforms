import time

from tqdm import tqdm  # type: ignore

from jl_transforms.run.hooks import Hook


class WarmupLoaderHook(Hook):
    def on_train_start(self, runner):
        for name, loader in runner.loaders.items():
            start = time.time()
            try:
                next(iter(loader))
            except StopIteration:
                tqdm.write(f"Loader '{name}' is empty on warmup.")

            duration = time.time() - start
            tqdm.write(f"Warmup for '{name}' loader took {duration:.2f}s")


class InputsTargetHook(Hook):
    def on_batch_start(self, runner):
        batch = runner.batch_state["raw"]
        inputs, targets = batch
        runner.batch_state.update(inputs=inputs, targets=targets)


class ToDeviceHook(Hook):
    def on_train_start(self, runner):
        runner.model.to(runner.device)

    def on_batch_start(self, runner):
        inputs = runner.batch_state["inputs"].to(runner.device)
        targets = runner.batch_state["targets"].to(runner.device)
        runner.batch_state.update(inputs=inputs, targets=targets)


class ComputeLossHook(Hook):
    def __init__(self, input_key: str = "logits", target_key: str = "targets"):
        self.input_key = input_key
        self.target_key = target_key

    def on_batch_end(self, runner):
        targets = runner.batch_state[self.target_key]
        logits = runner.batch_state[self.input_key]
        loss = runner.loss_fn(logits, targets)
        runner.batch_state.update(loss=loss)


class BackwardHook(Hook):
    def on_batch_end(self, runner):
        runner.optimizer.zero_grad()
        loss = runner.batch_state["loss"]
        loss.backward()
        runner.optimizer.step()


eval_hooks: list[Hook] = [
    WarmupLoaderHook(),
    InputsTargetHook(),
    ToDeviceHook(),
    ComputeLossHook(),
]

train_hooks: list[Hook] = [
    WarmupLoaderHook(),
    InputsTargetHook(),
    ToDeviceHook(),
    ComputeLossHook(),
    BackwardHook(),
]
