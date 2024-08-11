"""
Goal: We have an existing architecture from which we will copy all the forward() methods.
      We then reassign some of the fields in that architecture so that there is a mismatch between field classes.
      What we want is to then write a new architecture where those fields are replaced, and in classes that use that
      class, etc...
"""
from typing import Dict, Set, List, Type
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.nn import Module

import warnings

from .util import *

SUFFIX = "_modified"


class Node(ABC):

    def __init__(self):
        self.modified = False
        self.children: OrderedDict[str,Node] = OrderedDict()

    @abstractmethod
    def getClass(self) -> type:
        pass

    @abstractmethod
    def renderInitialiser(self) -> str:
        pass

    @abstractmethod
    def renderClass(self) -> str:
        pass

    def compareChildren(self, versus: "Node"):
        assert self.getClass() == versus.getClass()
        self.modified = False  # Reset since last comparison.

        fields1 = set(self.children)
        fields2 = set(versus.children)

        overlapping_fields = fields1 & fields2
        new_fields         = fields2 - fields1
        missing_fields     = fields1 - fields2  # These are not None-assigned fields. These are del'd fields. Realistically should be an empty set.

        # TODO: For ModuleList this has special implications.
        if missing_fields:
            warnings.warn(f"Note: the given module is missing submodule(s) {missing_fields}. That means .forward() likely won't work and needs to be reimplemented")
        if new_fields:
            warnings.warn(f"Note: the given module has new submodule(s) {new_fields}. Hence, .forward() doesn't use them, and needs to be reimplemented")

        # Find fields that now have a different underlying module class.
        replaced_fields   = {name for name in overlapping_fields if self.children[name].getClass() != versus.children[name].getClass()}
        unreplaced_fields = overlapping_fields - replaced_fields

        # If you found such a field: you were definitely altered.
        # For the other fields: it's possible that deep down they were altered, and hence they would want to print themselves. So, we recurse either way.
        for name in unreplaced_fields:
            self.children[name].compareChildren(versus.children[name])
        self.modified = len(replaced_fields) > 0 or any(child.modified for child in self.children.values())

    def renderClass_recursiveAndOnlyModified(self) -> str:
        """
        Render the class code for this class and all its submodules,
        but for each render, only output anything if the module was
        flagged as modified.
        """
        if not self.modified:  # Don't even need to check the children if they have been.
            return ""
        else:
            s = ""
            for _,child in self.children.items():
                child_string = child.renderClass_recursiveAndOnlyModified()
                if child_string:
                    child_string += "\n"
                s += child_string
            s += self.renderClass()
            return s


class NoneNode(Node):

    def getClass(self) -> type:
        return type(None)

    def renderInitialiser(self) -> str:
        return "None"

    def renderClass(self) -> str:
        return ""


class ModuleNode(Node):

    def __init__(self, reference: Module):
        super().__init__()
        self.reference = reference
        self.fillChildren()

    def fillChildren(self):
        for name, module in getNamedSubmodules(self.reference).items():
            self.children[name] = constructNode(module)

    def getClass(self) -> type:
        return self.reference.__class__

    def renderInitialiser(self) -> str:
        return f"{self.getClass().__name__}{SUFFIX if self.modified else ''}(config)"

    def renderClass(self) -> str:
        s = ""
        s += f"class {self.getClass().__name__}{SUFFIX}(nn.Module):\n\n"
        s += "    def __init__(config):\n"
        for name, child in self.children.items():
            s += f"        self.{name} = {child.renderInitialiser()}\n"
        s += "\n"

        # TODO: Copy .forward() from the original class.
        return alignCharacter(s, "=")


class NativeModuleNode(ModuleNode):
    """
    Modules whose constructor have an nn.
    """

    def renderInitialiser(self) -> str:
        return "nn." + self.getClass().__name__ + "(...)"

    def renderClass(self) -> str:
        return ""


class ModuleListNode(NativeModuleNode):
    """
    Special module node because you don't construct it with "ModuleList(config)"
    and because its modules don't come from field assignments but rather the constructor.
    """

    def renderInitialiser(self) -> str:
        warnings.warn("Rendering module list. Since we are unaware of where in the config the length of the list comes from, constants will be used.")

        # Find spans of the same class
        spans = []
        prev_class = None
        for _,child in self.children.items():
            if child.getClass() != prev_class:
                spans.append([])

            spans[-1].append(child)
            prev_class = child.getClass()

        # Render as multiple lists
        s = "nn.ModuleList("
        if len(spans) > 1:
            s += "\n            "

        rendered_lists = []
        for span in spans:
            rendered_lists.append(f"[{span[0].renderInitialiser()} for _ in range({len(span)})]")
        s += " +\\\n            ".join(rendered_lists)
        if len(spans) > 1:
            s += "\n        "
        s += ")"
        return s


def getNamedSubmodules(module: Module) -> Dict[str,Module]:
    """
    In torch.nn.Module, the __setattr__ method (which is called whenever you assign a field of an object) is overridden
    to divert all assignments of values that themselves have type torch.nn.Module to be added to a dictionary called
    self._modules rather than be added as actual fields.

    Indeed, when you call module.__dict__.keys(), you will only see generic names, no fields specific to the architecture.
    Those are hidden in the _modules field.
    """
    return module._modules


def constructNode(module: Module) -> Node:
    if module is None:
        return NoneNode()

    if module.__class__.__module__.startswith("torch.nn"):
        if module.__class__.__name__ == "ModuleList":
            return ModuleListNode(module)
        else:
            return NativeModuleNode(module)
    else:
        return ModuleNode(module)


def printDifference(instance: Module, klass: Type[Module]):
    reference_instance = klass(instance.config)

    tree1 = constructNode(instance)
    tree2 = constructNode(reference_instance)

    tree1.compareChildren(tree2)
    print(tree1.renderClass_recursiveAndOnlyModified())
