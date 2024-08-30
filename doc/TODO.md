# TODO
- Test if you can save_pretrained and load_pretrained a `ModelWithHead` subclass. I think saving is possible but loading may run into a FIXME we have left.
- Support for weight tying.
- Could eventually add support for a "multi-task head" that consists of a nested sequence of heads, returns a tuple of logits, weighs its losses, ... and uses the same model call for all of them. The way you would define it is likely by giving a sequence of `(ForTaskClass, HeadConfig, loss_weight)` tuples given that you need to build the losses.
