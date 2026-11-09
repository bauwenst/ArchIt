from archit.instantiation.heads import BiaffineTokenToTokenHeadConfig
from archit.util import dataclass_from_dict


def tst_deserialiseHeadConfig():
    head_config_on_disk = {'extended_model_config': {'layer_pooling': 1, 'stride': 256, 'word_pooling': '3'}, 'final_hidden_size_arcs': 500, 'final_hidden_size_relations': 100, 'head_dropout': 0.33, 'num_labels': 50, 'standardisation_exponent': 0}

    print(BiaffineTokenToTokenHeadConfig(**head_config_on_disk))  # <-- Not recursive
    print(dataclass_from_dict(BiaffineTokenToTokenHeadConfig, head_config_on_disk))  # <-- Recursive + Enum support
