
def test_equivalence_robertafortokens():
    """
    Test whether AutoModelForTokenClassification and ForSingleLabelTokenClassification make identical
    predictions after loading from the same RoBERTa checkpoint.
    """
    import copy
    import torch
    torch.set_printoptions(sci_mode=False)

    from transformers import AutoModelForTokenClassification, AutoTokenizer, RobertaModel
    from archit.instantiation.basemodels import RobertaBaseModel
    from archit.instantiation.tasks import ForSingleLabelTokenClassification, TokenClassificationHeadConfig

    model_archit = ForSingleLabelTokenClassification.from_pretrained("roberta-base", RobertaBaseModel, TokenClassificationHeadConfig(num_labels=19))
    model_hf     = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=19)

    # First of all: check that the encoders are equal.
    roberta_archit: RobertaModel = model_archit.model.core
    roberta_hf: RobertaModel     = model_hf.roberta
    assert torch.all(roberta_hf.embeddings.word_embeddings.weight == roberta_archit.embeddings.word_embeddings.weight).item()
    assert torch.all(roberta_hf.encoder.layer[5].output.dense.weight == roberta_archit.encoder.layer[5].output.dense.weight)

    # Synchronise heads
    model_archit.head.dense = copy.deepcopy(model_hf.classifier)
    assert torch.all(model_hf.classifier.weight == model_archit.head.dense.weight).item()

    # Input
    tk = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    batch = ["This is a very long sentence that has been artificially constructed just for the sake of it.",
             "A much shorter sentence with padding."]
    batch = tk(batch, return_tensors="pt", padding=True, truncation=True)

    # Inference
    model_archit.to("cuda")
    model_hf.to("cuda")
    batch = {k: v.to("cuda") for k,v in batch.items()}

    with torch.no_grad():
        logits1 = model_archit(**batch).logits
        logits2 = model_hf(**batch, return_dict=True).logits
        assert torch.all(logits1 == logits2).item()


if __name__ == "__main__":
    test_equivalence_robertafortokens()