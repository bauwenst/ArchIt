# ArchIt: <ins>A</ins>utomatic PyTo<ins>r</ins>ch ar<ins>chi</ins>tec<ins>t</ins>ures 
ArchIt lets you **put heads on top of language models** without having to re-implement heads per model or write dedicated `YourModelForThatTask` classes again and again.
It also helps you rewrite PyTorch code for base models augmented at runtime.

## `archit.instantiation`: Add any head to any base model
_Why in the heavens do there exist separate classes for 
[`BertForTokenClassification`](https://github.com/huggingface/transformers/blob/31d7392997c13162c58f2e3c6555b5e060d6f60b/src/transformers/models/bert/modeling_bert.py#L1234) and 
[`RobertaForTokenclassification`](https://github.com/huggingface/transformers/blob/31d7392997c13162c58f2e3c6555b5e060d6f60b/src/transformers/models/roberta/modeling_roberta.py#L1074)?_
The base model encodes tokens into embeddings. The head converts embeddings into logits. Which base model produces the embeddings,
and how it does so, is completely irrelevant to the head. Additionally, for all pairings of base model and head, the way
the base model feeds into the head is the exact same, and it is pointless to reimplement it in a new class.

In HuggingFace `transformers`, given _B_ base models and _T_ modelling tasks, you need _B x T_ model-task-specific
classes with _B x T_ head implementations. In ArchIt, there is only 1 model-task class (the `ModelWithHead`) which
can wrap any base model and any of the _T_ model-agnostic head implementations. _Implement once, use anywhere_.

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

| Transformers version | ArchIt version | 
|----------------------|----------------| 
| \<= v4.49.0          | \<= 2026.4.1   |
| \>= v5.22.1          | \> 2026.4.1    |

To install an older version of ArchIt, replace `YOUR_VERSION_HERE` below according to the table:
```shell
pip install "archit[all] == YOUR_VERSION_HERE"
```

## Usage
### `archit.instantiation`
You have some kind of model architecture that generates token embeddings (e.g. some variant of `RobertaModel`) and you 
want to put a head on it for fine-tuning. Because you are a sane individual, _you'd prefer not writing code for a head 
that has been defined hundreds of times before by others,_ and you also don't want to write a class for each 
model-head combination.

With ArchIt, you can just build the architecture at runtime:

```python
from transformers import RobertaConfig

from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.heads import BiaffineTokenToTokenHeadConfig
from archit.instantiation.tasks import ForDependencyParsing

model_with_head = ForDependencyParsing.fromModelAndHeadConfig(
    RobertaBaseModel(RobertaConfig()),
    BiaffineTokenToTokenHeadConfig()
)
```

"What if I have a pre-trained checkpoint of my core model?" No problem! The `from_pretrained` of these predefined 
model-with-head architectures will read your checkpoint and put the weights into the right parts of the model:

```python
from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.heads import BiaffineTokenToTokenHeadConfig
from archit.instantiation.tasks import ForDependencyParsing

model_with_head = ForDependencyParsing.from_pretrained(
    "path/to/core-checkpoint",
    RobertaBaseModel,
    BiaffineTokenToTokenHeadConfig()
)
```

All you need to give is your checkpoint, a wrapper to put around the specific implementation of your core embedding model,
and -- if the checkpoint is not already a checkpoint of `ForDependencyParsing` -- a config for the head you put on it.

### `archit.declaration`
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
