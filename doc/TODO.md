# TODO
- Test if you can save_pretrained and load_pretrained a `ModelWithHead` subclass. I think saving is possible but loading may run into a FIXME we have left.
- Test if the following can happen: 
  - Save an ArchIt ForToken model.
  - Load that checkpoint into an Archit ForCausal model. 
  - Because both have a `head.dense` parameter, I hypothesise that you will accidentally load the ForToken head into
    the causal head and make a mess.
- Support for weight tying.
- Could eventually add support for a "multi-task head" that consists of a nested sequence of heads, returns a tuple of logits, weighs its losses, ... and uses the same model call for all of them. The way you would define it is likely by giving a sequence of `(ForTaskClass, HeadConfig, loss_weight)` tuples given that you need to build the losses.
