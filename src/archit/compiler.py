"""
Goal: We have an existing architecture from which we will copy all the forward() methods.
      We then reassign some of the fields in that architecture so that there is a mismatch between field classes.
      What we want is to then write a new architecture where those fields are replaced, and in classes that use that
      class, etc...
"""
import warnings
from typing import Dict, Set

import torch
from torch.nn import Module
from transformers import AutoModel, RobertaModel

SUFFIX = "_modified"


def getNamedSubmodules(module: Module) -> Dict[str,Module]:
    """
    In torch.nn.Module, the __setattr__ method (which is called whenever you assign a field of an object) is overridden
    to divert all assignments of values that themselves have type torch.nn.Module to be added to a dictionary called
    self._modules rather than be added as actual fields.

    Indeed, when you call module.__dict__.keys(), you will only see generic names, no fields specific to the architecture.
    Those are hidden in the _modules field.
    """
    return module._modules


def recurse(module: torch.nn.Module, level=0):
    for key, mod in getNamedSubmodules(module):
        print("\t"*level, key, "is a", mod.__class__.__name__)
        recurse(mod, level+1)


def compareAndPrintDifferences(module1: torch.nn.Module, module2: torch.nn.Module, level=0) -> bool:
    m1 = getNamedSubmodules(module1)
    m2 = getNamedSubmodules(module2)

    m1_fields = set(m1.keys())
    m2_fields = set(m2.keys())

    overlapping_fields = m1_fields & m2_fields
    new_fields     = m2_fields - m1_fields
    missing_fields = m1_fields - m2_fields  # These are not None-assigned fields. These are del'd fields. Realistically should be an empty set.

    if missing_fields:
        warnings.warn(f"Note: the given module is missing submodule(s) {missing_fields}. That means .forward() likely won't work and needs to be reimplemented")
    if new_fields:
        warnings.warn(f"Note: the given module has new submodule(s) {new_fields}. Hence, .forward() doesn't use them, and needs to be reimplemented")

    # Find fields that now have a different class.
    replaced_fields = {name for name in overlapping_fields if m1[name].__class__ != m2[name].__class__}
    unreplaced_fields = overlapping_fields - replaced_fields

    # If you found such a field: you were definitely altered.
    # For the other fields: it's possible that deep down they were altered, and hence they would want to print themselves. So, we recurse either way.
    altered = len(replaced_fields) > 0
    unreplaced_but_modified = set()
    for name in unreplaced_fields:
        if compareAndPrintDifferences(m1[name], m2[name], level+1):
            altered = True
            unreplaced_but_modified.add(name)

    # All submodules have printed themselves. Time to print yourself if you were altered.
    if altered:
        printArchitecture(module2, unreplaced_but_modified)

    # Let your parent know it should print itself.
    return altered


def printArchitecture(module: Module, fields_with_generated_class: Set[str]):
    print(f"class {module.__class__.__name__}{SUFFIX}(nn.Module):")
    print()
    print("    def __init__(config):")
    for name, submod in getNamedSubmodules(module).items():
        print(f"        self.{name} = ", end="")
        if submod is None:
            print("None")
        elif submod.__class__.__module__.startswith("torch.nn"):  # PyTorch classes are assumed to be imported from torch.nn.
            if submod.__class__.__name__ == "ModuleList":
                # TODO: Add special support for ModuleList. Will be tricky if one of the layers has changed.
                pass
            else:
                print(f"nn.{submod.__class__.__name__}(config)")
        else:
            print(f"{submod.__class__.__name__}{SUFFIX if name in fields_with_generated_class else ''}(config)")
    print()
    # TODO: Copy .forward() from the original class.
