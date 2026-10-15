import sys
import types

import pytest

from camwatch.faces import onnxruntime_cuda_hint


def fake_versions(monkeypatch, ort_cuda, torch_cuda):
    build = types.SimpleNamespace(__version__="1.31.0", cuda_version=ort_cuda)
    capi = types.ModuleType("onnxruntime.capi")
    capi.build_and_package_info = build
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "onnxruntime.capi", capi)
    monkeypatch.setitem(sys.modules, "onnxruntime.capi.build_and_package_info", build)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(cuda=torch_cuda)))


def test_cuda13_onnxruntime_with_cuda12_torch_says_how_to_fix(monkeypatch):
    fake_versions(monkeypatch, "13.0", "12.6")
    hint = onnxruntime_cuda_hint()
    assert "built for CUDA 13 but torch uses CUDA 12" in hint and 'pip install "onnxruntime-gpu<1.27"' in hint


@pytest.mark.parametrize("ort_cuda, torch_cuda", [
    ("12.8", "12.6"),  # the supported pair: no hint
    ("13.0", None),    # CPU-only torch
    (None, "12.6"),    # CPU-only onnxruntime
])
def test_no_hint_when_nothing_to_fix(monkeypatch, ort_cuda, torch_cuda):
    fake_versions(monkeypatch, ort_cuda, torch_cuda)
    assert onnxruntime_cuda_hint() == ""
