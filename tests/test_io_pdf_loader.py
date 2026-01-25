import types

import caesar_ocr.io.loaders as loaders


def test_load_images_from_bytes_pdf(monkeypatch):
    dummy_image = types.SimpleNamespace(size=(123, 456), convert=lambda mode: types.SimpleNamespace(size=(123, 456)))

    def fake_convert_from_bytes(_bytes, dpi=300):
        return [dummy_image]

    monkeypatch.setattr(loaders, "convert_from_bytes", fake_convert_from_bytes)

    pages = loaders.load_images_from_bytes(b"%PDF-1.4")
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].width == 123
    assert pages[0].height == 456
