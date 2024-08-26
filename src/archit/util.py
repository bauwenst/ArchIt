from torch.nn import Module


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
