import types

import caesar_ocr.pipeline.analyze as analyze_mod


def test_pipeline_layoutlm_token_inference(monkeypatch):
    def fake_analyze_bytes(_bytes, lang="eng+deu"):
        class Dummy:
            doc_type = "unknown"
            predictions = []
            ocr_text = "Hello World"
            fields = {}
            tokens = [
                {"text": "Hello", "bbox": [0, 0, 1, 1], "page": 1},
                {"text": "World", "bbox": [0, 0, 1, 1], "page": 1},
            ]
            page_texts = ["Hello World"]

        return Dummy()

    def fake_load_images(_bytes, dpi=300):
        return [
            type("P", (), {"page": 1, "width": 100, "height": 200, "image": object()})(),
            type("P", (), {"page": 2, "width": 100, "height": 200, "image": object()})(),
        ]

    def fake_infer_tokens(_image, tokens, bboxes, model_dir):
        assert model_dir == "/tmp/model"
        if tokens == ["Hello", "World"]:
            return ["B-TEST", "I-TEST"], [0.9, 0.8]
        return [], []

    monkeypatch.setattr(analyze_mod, "analyze_bytes", fake_analyze_bytes)
    monkeypatch.setattr(analyze_mod, "load_images_from_bytes", fake_load_images)
    monkeypatch.setattr(analyze_mod, "infer_tokens", fake_infer_tokens)

    result = analyze_mod.analyze_document_bytes(b"dummy", layoutlm_token_model_dir="/tmp/model")
    data = result.to_dict()

    assert data["layoutlm_tokens"]["labels"] == ["B-TEST", "I-TEST"]
    assert data["layoutlm_tokens"]["scores"] == [0.9, 0.8]
    assert data["ocr"]["pages"][0]["tokens"][0]["label"] == "B-TEST"
    assert data["ocr"]["pages"][0]["tokens"][0]["label_score"] == 0.9
