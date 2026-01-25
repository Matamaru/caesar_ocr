import types

import caesar_ocr.layoutlm.infer as infer


def test_analyze_bytes_layoutlm_mock(monkeypatch):
    class DummyProcessor:
        def __call__(self, images=None, return_tensors=None, lang=None):
            class DummyTensor:
                def to(self, _device):
                    return self

            return {"pixel_values": DummyTensor()}

    class DummyModel:
        def __init__(self):
            self.config = types.SimpleNamespace(id2label={0: "A", 1: "B"})

        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, **_kwargs):
            class DummyLogits:
                def squeeze(self, _dim):
                    return self

            return types.SimpleNamespace(logits=DummyLogits())

    def fake_softmax(_logits, dim=-1):
        return types.SimpleNamespace(tolist=lambda: [0.2, 0.8])

    def fake_argmax(_logits, dim=-1):
        return types.SimpleNamespace(item=lambda: 1)

    monkeypatch.setattr(infer, "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyProcessor()))
    monkeypatch.setattr(infer, "AutoModelForSequenceClassification", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()))
    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(infer, "torch", types.SimpleNamespace(
        softmax=lambda logits, dim=-1: fake_softmax(logits, dim),
        argmax=lambda logits, dim=-1: fake_argmax(logits, dim),
        no_grad=lambda: DummyNoGrad(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setattr(infer, "load_images_from_bytes", lambda _b, dpi=300: [types.SimpleNamespace(image=object())])

    result = infer.analyze_bytes_layoutlm(b"dummy", model_dir="/tmp")
    assert result.doc_type == "B"
    assert result.label_id == 1
    assert result.scores["B"] == 0.8
