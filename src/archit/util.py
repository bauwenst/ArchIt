from typing import Tuple, Dict
from pathlib import Path
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel

from transformers.utils import cached_file, WEIGHTS_NAME
from transformers.modeling_utils import load_state_dict


def torchPrint(module: Module, scaffolding: str=":", truncation_depth: int=100):
    """
    Re-indent the PyTorch print with 4 spaces rather than 2 and also add scaffolding.
    """
    default_indent = "  "
    lines = repr(module).split("\n")

    lines_kept = []
    too_deep = False
    for line in lines:
        if line.startswith(default_indent*truncation_depth):
            if not too_deep:
                lines_kept.append(default_indent*truncation_depth + "...")
            too_deep = True
        else:
            lines_kept.append(line)
            too_deep = False

    print("\n".join(lines_kept).replace(default_indent, scaffolding + "   "))


def parameterCount(model: Module) -> Tuple[int,int]:
    trainable = 0
    total     = 0
    for p in model.parameters():
        n = p.numel()  # num(ber) el(ements)
        total += n
        if p.requires_grad:
            trainable += n

    return trainable, total


def parameterCountBaseVsHead(model: PreTrainedModel) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    base_trainable, base_total = parameterCount(model.base_model)
    all_trainable, all_total   = parameterCount(model)
    return (base_trainable, base_total), (all_trainable - base_trainable, all_total - base_total)


def checkpointToStateDict(checkpoint: str) -> Dict[str,Tensor]:
    """
    Very reduced version of the path resolution logic that happens inside
    transformers.modeling_utils.PreTrainedModel.from_pretrained.
    """
    if Path(checkpoint).is_dir():  # Local
        path = Path(checkpoint) / WEIGHTS_NAME
        if not path.is_file():
            raise RuntimeError(f"No file {WEIGHTS_NAME} in checkpoint folder {checkpoint}.")
        checkpoint = path.as_posix()
    else:  # Remote
        try:
            checkpoint = cached_file(path_or_repo_id=checkpoint, filename=WEIGHTS_NAME)
        except:
            raise RuntimeError(f"Could not find checkpoint {checkpoint}.")

    return load_state_dict(checkpoint)
