# TODO
- There is one very large technical debt that will require some redesign: we have linked *tasks* and *head architectures*,
  both in ArchIt and in LaMoTO, but this is not necessary at all. Sequence classification heads have an (implicit) *interface*
  they need to adhere to, but there are infinitely many `SequenceClassificationHead`s that can exist. We should really name
  heads after their architecture, like a `Pool_MLP_MLP_Head`, and then clearly, a task is not parameterised by a config for
  that head specifically. 
- Test if you can save_pretrained and load_pretrained a `ModelWithHead` subclass. I think saving is possible but loading may run into a FIXME we have left.
- Test if the following can happen: 
  - Save an ArchIt ForToken model.
  - Load that checkpoint into an Archit ForCausal model. 
  - Because both have a `head.dense` parameter, I hypothesise that you will accidentally load the ForToken head into
    the causal head and make a mess.
- Support for weight tying.
- Could eventually add support for a "multi-task head" that consists of a nested sequence of heads, returns a tuple of logits, weighs its losses, ... and uses the same model call for all of them. The way you would define it is likely by giving a sequence of `(ForTaskClass, HeadConfig, loss_weight)` tuples given that you need to build the losses.
