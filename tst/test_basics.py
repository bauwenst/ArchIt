from transformers import RobertaForMaskedLM
from archit import printDifference


def test_none():
    model: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.roberta.embeddings = None

    printDifference(model, RobertaForMaskedLM)


def test_module():
    class DummyModule:
        pass

    model: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.roberta.embeddings = DummyModule()

    printDifference(model, RobertaForMaskedLM)


def test_modulelist():
    class DummyModule:
        pass

    model: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.roberta.encoder.layer[4] = DummyModule()

    printDifference(model, RobertaForMaskedLM)


if __name__ == "__main__":
    # test_none()
    # test_module()
    test_modulelist()
