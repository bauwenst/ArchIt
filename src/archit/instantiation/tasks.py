"""
Links 1. which head belongs to which task and 2. which loss belongs to which task and how it is computed from logits and labels.
"""
from torch import Tensor, FloatTensor
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from .abstracts import ModelWithHead, BaseModel
from .heads import TokenClassificationHead, TokenClassificationHeadConfig, SequenceClassificationHeadConfig, SequenceClassificationHead


class ForSingleLabelTokenClassification(ModelWithHead):

    def __init__(self, base_model: BaseModel, head_config: TokenClassificationHeadConfig):
        super().__init__(
            base_model,
            TokenClassificationHead(base_model.convertConfig(), head_config),
            CrossEntropyLoss()
        )
        self.num_labels = head_config.num_labels

    def getLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))


class ForSingleLabelSequenceClassification(ModelWithHead):
    """
    For datasets where each example text has one integer as label, between 0 and num_labels-1.
    """

    def __init__(self, base_model: BaseModel, head_config: SequenceClassificationHeadConfig):
        super().__init__(
            base_model,
            SequenceClassificationHead(base_model.convertConfig(), head_config),
            CrossEntropyLoss()
        )
        self.num_labels = head_config.num_labels

    def getLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))


class ForMultiLabelSequenceClassification(ModelWithHead):
    """
    For datasets where each example text has a list of boolean values as label, of length num_labels.
    """

    def __init__(self, base_model: BaseModel, head_config: SequenceClassificationHeadConfig):
        super().__init__(
            base_model,
            SequenceClassificationHead(base_model.convertConfig(), head_config),
            BCEWithLogitsLoss()
        )
        self.num_labels = head_config.num_labels

    def getLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        return self.loss_function(logits, labels)


class ForSequenceRegression(ModelWithHead):

    def __init__(self, base_model: BaseModel, head_config: SequenceClassificationHeadConfig):
        super().__init__(
            base_model,
            SequenceClassificationHead(base_model.convertConfig(), head_config),
            MSELoss()
        )
        self.num_labels = head_config.num_labels

    def getLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        labels = labels.to(logits.device)
        if self.num_labels == 1:  # MISO regression. Label is just a number 4.20, and if it's a one-number list [4.20], we squeeze away that list dimension.
            return self.loss_function(logits.squeeze(), labels.squeeze())
        else:  # MIMO regression. Label is a list [4.20, 6.90]. (Technically you could also squeeze these, since it won't do anything. Not sure why HF separates these two cases.)
            return self.loss_function(logits, labels)
