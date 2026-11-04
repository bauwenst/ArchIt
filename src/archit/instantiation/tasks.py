"""
Links (1) which head belongs to which task and (2) which loss belongs to which task and how it is computed from logits and labels.
"""
import torch
from torch import Tensor, FloatTensor
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, _Loss

from .configs import PC, HC
from .abstracts import *
from .extensions import ForTokensGroupedByWord, ForNestedBatches
from .heads import *
from .heads import _LMHead


class ForSingleLabelTokenClassification(ModelWithHead[PC,TokenClassifierHeadConfig]):

    @classmethod
    def headSignature(cls):
        return TokenClassifierHeadConfig

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.head_config.num_labels), labels.view(-1))


class ForSingleLabelSequenceClassification(ModelWithHead[PC,PooledClassifierHeadConfig]):
    """
    For datasets where each example text has one integer as label, between 0 and num_labels-1.
    """

    @classmethod
    def headSignature(cls):
        return PooledClassifierHeadConfig

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.head_config.num_labels), labels.view(-1))  # .view(-1) is equivalent to flattening the tensor, and .view(-1, classes) flattens every dimension except the one at the end that is as big as the amount of classes.


class ForMultiLabelSequenceClassification(ModelWithHead[PC,PooledClassifierHeadConfig]):
    """
    For datasets where each example text has a list of boolean values as label, of length num_labels.
    """

    @classmethod
    def headSignature(cls):
        return PooledClassifierHeadConfig

    def buildLoss(self) -> _Loss:
        return BCEWithLogitsLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits, labels)


class ForSequenceRegression(ModelWithHead[PC,PooledClassifierHeadConfig]):

    @classmethod
    def headSignature(cls):
        return PooledClassifierHeadConfig

    def buildLoss(self) -> _Loss:
        return MSELoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        if self.config.head_config.num_labels == 1:  # MISO regression. Label is just a number 4.20, and if it's a one-number list [4.20], we squeeze away that list dimension.
            return self.loss_function(logits.squeeze(), labels.squeeze())
        else:  # MIMO regression. Label is a list [4.20, 6.90]. (Technically you could also squeeze these, since it won't do anything. Not sure why HF separates these two cases.)
            return self.loss_function(logits, labels)


class ForExtractiveQA(ModelWithHead[PC,TokenClassifierHeadConfig]):

    @classmethod
    def headSignature(cls):
        return TokenClassifierHeadConfig

    @classmethod
    def adjustHeadConfig(cls, head_config: TokenClassifierHeadConfig):
        head_config.num_labels = 2

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)

        # Disentangle QA-start and QA-end logits. Squeeze and (for some reason) make contiguous.
        start_logits, end_logits = logits.split(1, dim=-1)  # "split the last dimension into groups of 1"
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits  .squeeze(-1).contiguous()

        # Disentangle QA-start and QA-end labels. Squeeze when needed. No contiguous needed apparently.
        start_positions, end_positions = labels.split(1, dim=-1)  # B' x 2  ->  B' x 1 and B' x 1
        if len(start_positions.size()) > 1:  # # If we are on multi-GPU, split add a dimension.
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # Sometimes the start/end positions are outside the input length; we truncate those back to the final index of the input.
        L = start_logits.size(1)
        start_positions = start_positions.clamp(0, L)
        end_positions   = end_positions  .clamp(0, L)

        # Get loss as average of start and end loss
        self.loss_function.ignore_index = L
        start_loss = self.loss_function(start_logits, start_positions)
        end_loss   = self.loss_function(end_logits, end_positions)
        return (start_loss + end_loss) / 2


class ForExtractiveAQA(ModelWithHead[PC,OneTokenHeadOnePooledHeadConfig]):
    """
    Architecture for the task of predicting whether a given question can be answered given a context,
    and if so, which span of the context contains the answer.
    """

    @classmethod
    def headSignature(cls):
        return OneTokenHeadOnePooledHeadConfig

    @classmethod
    def adjustHeadConfig(cls, head_config: OneTokenHeadOnePooledHeadConfig):
        head_config.token_head.num_labels  = 2
        head_config.pooled_head.num_labels = 2

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: tuple[Tensor,Tensor], labels: tuple[Tensor,Tensor]) -> FloatTensor:
        # The head always produces both QA and unanswerability (UA) logits for all examples, but:
        #   - The thresholded prediction of whether each example is answerable according to the model has NO influence
        #     on the outcome of this method. Downstream code may want to prevent doing QA for examples that are predicted
        #     to be UA, but that prediction has no power here.
        #   - The hard UA labels are used for UA loss AND are used to decide which QA logits are counted towards QA loss.

        # Disentangle QA and UA
        qa_logits, ua_logits = logits  # qa_logits is B x L x 2 (equivalent of B x L and B x L) where the last dimension has a logit for a token being the start and the end, each separately normalised across L. ua_logits is B x 2, normalised across that 2.
        qa_labels, ua_labels = labels  # qa_labels is B x 2 (equivalent of B x 1 and B x 1). ua_labels is B x 1.

        # Send to device
        qa_labels, ua_labels = qa_labels.to(qa_logits.device), ua_labels.to(ua_logits.device)

        # Mask out QA logits and labels for examples that are unanswerable. This is only worth it when you're going to have QA loss.
        # Note that the same type of masking will likely be done during evaluation, but then based on thresholded predictions. This
        # never happens in this method because this method computes loss for the heads and it makes no sense to punish the QA heads
        # with an arbitrarily high loss on examples we know they can't handle.
        if ua_labels is not None and qa_labels is not None:
            mask = torch.where(ua_labels)  # True when answerable. Say there are B' <= B of these.
            qa_logits = qa_logits[mask]
            qa_labels = qa_labels[mask]

        # Now, like QA: disentangle QA-start and QA-end logits, squeeze and (for some reason) make contiguous.
        start_logits, end_logits = qa_logits.split(1, dim=-1)  # B' x L x 2  ->  B' x L and B' x L.
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits  .squeeze(-1).contiguous()

        # 1. QA loss
        qa_loss = None
        if qa_labels is not None:
            # Disentangle QA-start and QA-end labels. Squeeze when needed. No contiguous needed apparently.
            start_positions, end_positions = qa_labels.split(1, dim=-1)  # B' x 2  ->  B' x 1 and B' x 1
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # Sometimes the start/end positions are outside the input length; we truncate those back to the final index of the input.
            L = start_logits.size(1)
            start_positions = start_positions.clamp(0, L)
            end_positions   = end_positions  .clamp(0, L)

            # Get loss as average of start and end loss
            self.loss_function.ignore_index = L
            start_loss = self.loss_function(start_logits, start_positions)
            end_loss   = self.loss_function(end_logits, end_positions)
            qa_loss = (start_loss + end_loss)/2
            self.loss_function.ignore_index = -100

        # 2. UA loss
        ua_loss = None
        if ua_labels is not None:
            ua_loss = self.loss_function(ua_logits.view(-1, 2), ua_labels.long().view(-1))

        # 3. Compose the losses.
        if qa_loss is not None and ua_loss is not None:
            total_loss = qa_loss + self.config.head_config.pooled_loss_weight * ua_loss
        elif qa_loss is not None:
            total_loss = qa_loss
        elif ua_loss is not None:
            total_loss = ua_loss  # No weight since it is the only term.
        else:
            total_loss = None

        return total_loss


class ForSingleAnswerMultipleChoice(ForNestedBatches[PC,PooledClassifierHeadConfig]):
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

    @classmethod
    def headSignature(cls):
        return TwoLayerPoolingHeadForNestedBatches

    @classmethod
    def adjustHeadConfig(cls, head_config: PooledClassifierHeadConfig):
        if head_config.num_labels != 1:
            raise NotImplementedError("Although multi-task multiple choice is technically possible, it isn't supported currently.")

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        """
        :param logits: B x C x 1 tensor (B examples of C choices, each scored on 1 task) or B x C tensor.
        :param labels: B x 1 tensor or B vector.
        """
        logits, labels = logits.squeeze(), labels.squeeze()
        labels = labels.to(logits.device)
        return self.loss_function(logits, labels)


class _ForLanguageModelling(ModelWithHead[PC,TokenVocabularyHeadConfig]):

    @classmethod
    def headSignature(cls):
        return TokenVocabularyHeadConfig

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    # Override PretrainedModel instance methods to support weight tying. We do it here because ModelWithHead has access to both the base model and the head.

    def get_expanded_tied_weights_keys(self, all_submodels: bool=False) -> dict[str,str]:  # We implement tied weights by dynamically looking up the output and input embeddings.
        if all_submodels:  # Recursive case
            return super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        else:  # Base case
            if self.config.head_config.tie_to_model:  # Finally, the generics system pays off.
                return {self._get_name_of_module(self.get_output_embeddings()): self._get_name_of_module(self.get_input_embeddings())}
            else:
                return {}

    def get_input_embeddings(self) -> nn.Module:
        return super().get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        super().set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        assert isinstance(self.head, _LMHead)
        return self.head.get_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        assert isinstance(self.head, _LMHead)
        self.head.set_embeddings(new_embeddings)


class ForCausalLM(_ForLanguageModelling[PC]):

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)

        # We are doing next-token prediction; shift prediction scores and input ids by one.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        return self.loss_function(shifted_logits.view(-1, self.config.base_model_config.vocab_size), shifted_labels.view(-1))


class ForMaskedLM(_ForLanguageModelling[PC]):

    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.config.base_model_config.vocab_size), labels.view(-1))


class ForDependencyParsing(ForTokensGroupedByWord[PC,TokenToTokenDualHeadsConfig]):
    """
    For the task of dependency parsing, based on Supar.

    Note that Supar's BiaffineDependencyModel represents a model-with-head, but has an unworkable interface:
        1. The head is defined inside a subclass of the base model. The super constructor makes a guaranteed call to AutoModel.
        2. The forward() method returns logits and no loss. It has a separate method for loss which is called by its users, which is not how you do that.
    """

    def __init__(self, combined_config: CombinedConfig[PC,TokenToTokenDualHeadsConfig], base_model_class: type[BaseModel[PC]]):
        super().__init__(combined_config, combined_config.head_config.extended_model_config, base_model_class)  # TODO: Yeah that's kind of a problem

    @classmethod
    def headSignature(cls):
        return TokenToTokenDualHeadsConfig

    def buildLoss(self) -> _Loss:
        return CrossEntropyLoss()

    def computeLoss(self, logits: tuple[Tensor,Tensor], labels: tuple[Tensor,Tensor]) -> FloatTensor:
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
