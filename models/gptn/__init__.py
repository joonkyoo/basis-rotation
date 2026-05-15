from .stage_first import StageFirst
from .stage_mid import StageMid
from .stage_last import StageLast

def arch():
    return "gptn"

def model(criterion, vocab_size, block_size, dropout=0.0, n_layer=32, n_head=6, n_embd=384):
    n_layer_mid = 1
    n_layer_mid_last = 1

    modules = [
        (StageFirst(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=1), ["input0"], ["out0"])
    ]

    for i in range(1, n_layer - 2):
        modules.append((StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), [f"out{i-1}"], [f"out{i}"]))

    modules.append((StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid_last), [f"out{n_layer - 3}"], [f"out{n_layer - 2}"]))
    modules.append((StageLast(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=1), [f"out{n_layer - 2}"], ["output"]))
    modules.append((criterion, ["output"], ["loss"]))
    return modules
