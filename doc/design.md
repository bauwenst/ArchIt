# Design notes
## Abstracts
### BaseModel
#### Why is BaseModelExtended not a BaseModel?
- Has been documented in its docstring.

## Mixins
### Stateful loss
#### Why not have the BaseModel's extra loss terms be delivered via the BaseModelOutput?
- This would require the loss terms to be known at the *end* of BaseModel's inference, i.e. in its final module, and hence
  every single module and its output would have to be altered.

#### Why ask the model developer to extend a class mixin rather than adding a method (and base class) post-hoc like HfMultitaskTrainer?
- The developer either way has to change their model code.
- Patching prevents autocompletion of the added methods. It's completely against OOP principles.
- Rather than having the model check "Do I have this method to communicate with the trainer?" you can just as well check
  whether one of your fields has a value that indicates you do.

#### Ignoring loss
Some base models produce loss terms, but we don't always want to include them (e.g. in fine-tuning). How do we make it
so the developer easily provides this functionality to the user?
- You could set the extra loss term's coefficient to 0.
  - Con: this requires altering the model config since this is where
    that coefficient resides.
  - Con: you still store a computation graph, just with a multiplied-by-0 node added.
- You could let the BaseModel have None in its loss field rather than a loss object, so that it also registers None in
  the underlying modules.
  - Con: the developer has to alter the constructor when extending BaseModel.
  - Con: computeLoss() would have to do a field-is-None check. Fields shouldn't be optionals.
- You could let the developer override computeLoss() to return a zero tensor rather than the loss state.
  - Con: you still store a computation graph.
  - Con: the developer would have to know he still has to clear the loss state to prevent a memory leak.
- You could let the BaseModel register None in the model's modules instead of a reference to its loss state.
  - Pro: no computation graph stored, hence no memory leak.
  - How does the developer cause this None to be registered rather than the BaseModel's loss state field? 
    - Override a boolean method to return False, and then... 
      - You could choose between None or the loss state when deciding which value to register.
      - You could store an "enabled" flag in the loss state that disables .add() calls.
- You could let the BaseModel not do any registering.
  - Pro: no computation graph, no memory leak.
  - Pro: the developer indicates *whether* the BaseModel registers the loss state by overriding a method that says
         *how* to register the loss state.

We choose the latter approach.
