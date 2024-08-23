# ArchIt: <ins>A</ins>utomatic PyTo<ins>r</ins>ch ar<ins>chi</ins>tec<ins>t</ins>ures 
ArchIt helps you rewrite PyTorch code for base models augmented at runtime, and lets you put heads on top of models without 
having to write dedicated task classes again and again.

## `archit.declaration`: Convert a PyTorch instance into PyTorch architecture classes.
Recursively rewrite a class hierarchy (i.e. generate Python code of PyTorch architectures) so that in-place modifications
are now defined explicitly.

As an example of this: I'm involved in two projects where I replace the embedding matrix of a `RobertaForMaskedLM` by a 
new class. If I want to load a checkpoint of that model, I need to write a new class definition for the `RobertaEmbeddings` 
that uses my replacement of the `Embedding`, a new `RobertaModel` using the new embeddings, and a new `RobertaForMaskedLM` 
using that new model. ArchIt writes that code for you.

## `archit.instantiation`: Add heads to a base model, without needing to write `YourModelForThatTask` classes
Why in the heavens do we need separate classes for `RobertaForTokenClassification` and `DebertaForTokenclassification`?
The base model encodes tokens into embeddings, and the head, which only cares about the resulting embeddings, converts
them to logits. Separation of concerns. There is no need to rewrite "model-with-head" classes over and over again for each
model augmentation.

## Installation
```shell
pip install "archit @ git+https://github.com/bauwenst/ArchIt"
```

## Usage
Minimal working example to show what ArchIt does:
```python
from transformers import RobertaForMaskedLM
import torch

class CoolNewEmbeddingMatrix(torch.nn.Module):
    def forward(self, input_ids):
        pass

model_with_head = RobertaForMaskedLM.from_pretrained("roberta-base")
model_with_head.roberta.embeddings.word_embeddings = CoolNewEmbeddingMatrix()
# ^--- This works, but there is no class definition declaring word_embeddings as a CoolNewEmbeddingMatrix.


from archit import printDifference
printDifference(model_with_head, RobertaForMaskedLM)  # Outputs Python code for 3 new classes.
```