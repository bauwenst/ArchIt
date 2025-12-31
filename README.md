# ArchIt: <ins>A</ins>utomatic PyTo<ins>r</ins>ch ar<ins>chi</ins>tec<ins>t</ins>ures 
ArchIt lets you put heads on top of models without having to write dedicated task classes again and again.
It also helps you rewrite PyTorch code for base models augmented at runtime.

_**NOTICE:** from Transformers v4.50.0 onwards, the implementation of `from_pretrained` was [changed so significantly](https://github.com/huggingface/transformers/commit/071a161d3e38f56dbda2743b979f0afeed2cd4f1)
that ArchIt is broken for all versions beyond `v4.49.0`. This will be fixed eventually. See the table below._

## `archit.instantiation`: Add heads to a base model, without needing to write `YourModelForThatTask` classes
Why in the heavens do we need separate classes for `RobertaForTokenClassification` and `DebertaForTokenclassification`?
The base model encodes tokens into embeddings, and the head, which only cares about the resulting embeddings, converts
them to logits. Separation of concerns. There is no need to rewrite "model-with-head" classes over and over again for each
model augmentation.

This part of ArchIt is the backbone behind the [LaMoTO](https://github.com/bauwenst/LaMoTO) package.

## `archit.declaration`: Convert a PyTorch instance into PyTorch architecture classes.
Recursively rewrite a class hierarchy (i.e. generate Python code of PyTorch architectures) so that in-place modifications
are now defined explicitly.

As an example of this: I'm involved in two projects where I replace the embedding matrix of a `RobertaForMaskedLM` by a 
new class. If I want to load a checkpoint of that model, I need to write a new class definition for the `RobertaEmbeddings` 
that uses my replacement of the `Embedding`, a new `RobertaModel` using the new embeddings, and a new `RobertaForMaskedLM` 
using that new model. ArchIt writes that code for you.

## Installation
Due to severe implementational changes in the `transformers` package, the version of ArchIt you need
depends on which version of `transformers` you are on:

| Transformers version | ArchIt version | ArchIt commit |
|----------------------|----------------|---------------| 
| \<= v4.49.0          | \<= 2026.01.01 | 2ae3c29       |
| \>= v4.50.0          | \>= 2026.01.02 | ???           | 

To install a specific version, replace `YOUR_COMMIT_HERE` by the relevant ArchIt commit.
```shell
pip install "archit @ git+https://github.com/bauwenst/ArchIt.git@YOUR_COMMIT_HERE"
```

## Usage
### Instantiation
You have some kind of model architecture that generates token embeddings (e.g. some variant of `RobertaModel`) and you 
want to put a head on it for fine-tuning. Because you are a sane individual, you'd prefer not writing code for a head 
that has been defined hundreds of times before by others, and you also don't want to write a class for each 
model-head combination.

With ArchIt, you can just build the architecture at runtime:
```python
from transformers import RobertaConfig

from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.tasks import ForDependencyParsing
from archit.instantiation.heads import DependencyParsingHeadConfig

model_with_head = ForDependencyParsing.fromModelAndHeadConfig(
    RobertaBaseModel(RobertaConfig()),
    DependencyParsingHeadConfig()
)
```

"What if I have a pre-trained checkpoint of my core model?" No problem! The `from_pretrained` of these predefined 
model-with-head architectures will read your checkpoint and put the weights into the right parts of the model:
```python
from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.tasks import ForDependencyParsing
from archit.instantiation.heads import DependencyParsingHeadConfig

model_with_head = ForDependencyParsing.from_pretrained(
    "path/to/core-checkpoint", 
    RobertaBaseModel, 
    DependencyParsingHeadConfig()
)
```
All you need to give is your checkpoint, a wrapper to put around the specific implementation of your core embedding model,
and -- if the checkpoint is not already a checkpoint of `ForDependencyParsing` -- a config for the head you put on it.

### Declaration
You've defined a PyTorch architecture in code. Now you reassign one of its fields.
This new PyTorch architecture exists in memory, but not in code.

With ArchIt, the code for the modified architecture can be generated automatically:
```python
from transformers import RobertaForMaskedLM
import torch

class CoolNewEmbeddingMatrix(torch.nn.Module):
    def forward(self, input_ids):
        pass

model_with_head = RobertaForMaskedLM.from_pretrained("roberta-base")
model_with_head.roberta.embeddings.word_embeddings = CoolNewEmbeddingMatrix()
# ^--- This works, but there is no class definition declaring word_embeddings as a CoolNewEmbeddingMatrix.

from archit.declaration.compiler import printDifference
printDifference(model_with_head, RobertaForMaskedLM)  # Outputs Python code for 3 new classes.
```