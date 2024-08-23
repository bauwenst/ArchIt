from torch.nn import Module


def torchPrint(module: Module, scaffolding: str=":"):
    """
    Re-indent the PyTorch print with 4 spaces rather than 2 and also add scaffolding.
    """
    print(repr(module).replace("  ", scaffolding + "   "))
