# ArchIt: convert a PyTorch instance into PyTorch architecture classes.
Recursively rewrite a class hierarchy (i.e. generate Python code of PyTorch architectures) so that in-place modifications
are now defined explicitly.

As an example of this: I have two projects running where I replace the embedding matrix of a `RobertaForMaskedLM` by a 
new class. If I want to load a checkpoint of that model, I need to write a new class definition for the `RobertaEmbeddings` 
that uses my replacement of the `Embedding`, a new `RobertaModel` using the new embeddings, and a new `RobertaForMaskedLM` 
using that new model. ArchIt writes that code for you.

## Installation
```shell
pip install "archit @ git+https://github.com/bauwenst/ArchIt.git"
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