"""
Links (1) which head belongs to which task and (2) which loss belongs to which task and how it is computed from logits and labels.
"""
import torch
from torch import Tensor, FloatTensor
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, _Loss

from .configs import PC
from .abstracts import *
from .abstracts import Tensors
from .extensions import ForTokensGroupedByWord, ForNestedBatches
from .heads import *


class ForSingleLabelTokenClassification(ModelWithHead[PC,TokenClassificationHeadConfig]):

    @classmethod
    @property
    def head_class(cls):
        return TokenClassificationHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.head_config.num_labels), labels.view(-1))


class ForSingleLabelSequenceClassification(ModelWithHead[PC,SequenceClassificationHeadConfig]):
    """
    For datasets where each example text has one integer as label, between 0 and num_labels-1.
    """

    @classmethod
    @property
    def head_class(cls):
        return SequenceClassificationHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.head_config.num_labels), labels.view(-1))  # .view(-1) is equivalent to flattening the tensor, and .view(-1, classes) flattens every dimension except the one at the end that is as big as the amount of classes.


class ForMultiLabelSequenceClassification(ModelWithHead[PC,SequenceClassificationHeadConfig]):
    """
    For datasets where each example text has a list of boolean values as label, of length num_labels.
    """

    @classmethod
    @property
    def head_class(cls):
        return SequenceClassificationHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return BCEWithLogitsLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits, labels)


class ForSequenceRegression(ModelWithHead[PC,SequenceClassificationHeadConfig]):

    @classmethod
    @property
    def head_class(cls):
        return SequenceClassificationHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return MSELoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        if self.config.head_config.num_labels == 1:  # MISO regression. Label is just a number 4.20, and if it's a one-number list [4.20], we squeeze away that list dimension.
            return self.loss_function(logits.squeeze(), labels.squeeze())
        else:  # MIMO regression. Label is a list [4.20, 6.90]. (Technically you could also squeeze these, since it won't do anything. Not sure why HF separates these two cases.)
            return self.loss_function(logits, labels)


class ForExtractiveQA(ModelWithHead[PC,ExtractiveQAHeadConfig]):

    @classmethod
    @property
    def head_class(cls):
        return ExtractiveQAHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        start_logits, end_logits = logits.split(1, dim=-1)  # "split the last dimension into groups of 1"
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits.squeeze(-1).contiguous()

        labels = labels.to(logits.device)
        start_positions, end_positions = labels.split(1, dim=-1)

        # If we are on multi-GPU, split add a dimension.
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # Sometimes the start/end positions are outside our model inputs, we ignore these terms.
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions   = end_positions.clamp(0, ignored_index)

        self.loss_function.ignore_index = ignored_index
        start_loss = self.loss_function(start_logits, start_positions)
        end_loss   = self.loss_function(end_logits, end_positions)
        return (start_loss + end_loss) / 2


class ForSingleAnswerMultipleChoice(ForNestedBatches[PC,SequenceClassificationHeadConfig]):
    """
    Architecture for tasks of the form (question, answer1, answer2, ...) which sends each question-answer pair through
    the model like
        (question, answer1) -> logit1
        (question, answer2) -> logit2
        ...
    and then does a softmax and cross-entropy across the logits to maximise the labelled choice and minimise the rest.
    There are variations on this problem that are not supported currently:
        - Multi-task multiple choice: say you have a model that tries to answer questions like an oracle on the one hand
                                      (factual labels), and like the general population on the other (survey labels).
                                      This is doable by having 2 logits per (question, answer) pair, generated by a 2-label
                                      head, and softmaxing over each sequence of logits separately.
                                      This requires a fixed-length list of labels per example, rather than a single label.
        - Multi-answer multiple choice: for questions like "select all true statements". This is more like a multi-label
                                        question and requires BCE loss across the logits. Requires a variable-length list of labels per example.
                                        (If combined with multi-task, you need a fixed-length list of variable-length lists of labels.)
        - Any-answer multiple choice: similar to multi-answer except you get full marks when you select at least one of
                                      the true answers. Gradient descent would be the same as BCE loss; metrics would be different.
    """

    def __init__(self, combined_config: CombinedConfig[PC,SequenceClassificationHeadConfig], model: BaseModel[PC], head: Head[SequenceClassificationHeadConfig], loss: _Loss):
        super().__init__(combined_config=combined_config, model=model, head=head, loss=loss)
        assert isinstance(head, SequenceClassificationHeadForNestedBatches)  # This is already asserted by the constructor. We do it here just for type checking.
        if head.dense2.weight.shape[0] != 1:  # Note that nn.Linear stores weights in the form out x in.
            raise NotImplementedError("Although multi-task multiple choice is technically possible, it isn't supported currently.")

    @classmethod
    @property
    def head_class(cls):
        return SequenceClassificationHeadForNestedBatches

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        """
        :param logits: B x C x 1 tensor (B examples of C choices, each scored on 1 task) or B x C tensor.
        :param labels: B x 1 tensor or B vector.
        """
        logits, labels = logits.squeeze(), labels.squeeze()
        labels = labels.to(logits.device)
        return self.loss_function(logits, labels)


class ForCausalLM(ModelWithHead[PC,CausalLMHeadConfig]):

    @classmethod
    @property
    def head_class(cls):
        return CausalLMHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)

        # We are doing next-token prediction; shift prediction scores and input ids by one.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        return self.loss_function(shifted_logits.view(-1, self.config.base_model_config.vocab_size), shifted_labels.view(-1))


class ForMaskedLM(ModelWithHead[PC,MaskedLMHeadConfig]):

    @classmethod
    @property
    def head_class(cls):
        return MaskedLMHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.base_model_config.vocab_size), labels.view(-1))


class ForDependencyParsing(ForTokensGroupedByWord[PC, DependencyParsingHeadConfig]):
    """
    For the task of dependency parsing, based on Supar.

    Note that Supar's BiaffineDependencyModel represents a model-with-head, but has an unworkable interface:
        1. The head is defined inside a subclass of the base model. The super constructor makes a guaranteed call to AutoModel.
        2. The forward() method returns logits and no loss. It has a separate method for loss which is called by its users, which is not how you do that.
    """

    def __init__(self, combined_config: CombinedConfig[PC,DependencyParsingHeadConfig], model: BaseModel[PC], head: Head[DependencyParsingHeadConfig], loss: _Loss):
        super().__init__(combined_config, combined_config.head_config.extended_model_config, model, head, loss)

    @classmethod
    @property
    def head_class(cls):
        return DependencyParsingHead

    @classmethod
    def buildLoss(cls) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensors, labels: Tensors) -> FloatTensor:
        arc_scores, rel_scores = logits
        arc_labels, rel_labels = labels

        device = arc_scores.device
        mask = torch.ones((arc_labels.size(0), arc_labels.size(1)), dtype=torch.bool, device=device)  # You can simply initialise the mask to a full-1 matrix, since the data preprocessor already put a -100 on all padding and special tokens. In Supar, the mask is supplied, and modified to explicitly mask out the first token of each sentence (mask[:, 0] = 0), probably because they're assuming "[BoS] tokens [EoS]" format, i.e. only two special tokens, where the EoS is a dummy for the tree root's head. Your data collator should handle special tokens when it constructs the labels already!
        mask = mask & arc_labels.ge(0)  # From this line onward, the implementation is identical to Supar's BiaffineModel.loss() method with partial=True.

        # Mask and send to device.
        arc_scores, rel_scores = arc_scores[mask], rel_scores[mask]
        arc_labels, rel_labels = arc_labels[mask].to(device), rel_labels[mask].to(device)
        rel_scores = rel_scores[torch.arange(len(arc_labels)), arc_labels]
        return self.loss_function(arc_scores, arc_labels) + self.loss_function(rel_scores, rel_labels)
