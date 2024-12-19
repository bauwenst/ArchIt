from typing import Tuple, Dict
from enum import Enum
from pathlib import Path
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel

from dataclasses import fields
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


def dataclass_from_dict(cls: type, as_dict: dict):
    """
    Based on https://stackoverflow.com/a/68395388/9352077
    """
    # Base case: raise TypeError when cls is not a dataclass type.
    field_objects = fields(cls)

    # Recursive case
    field_types_lookup = {field.name: field.type for field in field_objects}
    kwargs = {}
    for field_name, value in as_dict.items():
        try:
            expected_type = field_types_lookup[field_name]
        except KeyError:
            # Key in the source is not a field of the class. Keep this excess value in the kwargs; it won't end up as a field, though.
            kwargs[field_name] = value
            continue

        try:
            kwargs[field_name] = dataclass_from_dict(expected_type, value)
        except TypeError:
            # Not a dataclass. If the type is an enum, do still deserialise it.
            if issubclass(expected_type, Enum):
                kwargs[field_name] = expected_type(value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)
