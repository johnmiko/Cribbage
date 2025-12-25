import importlib
import sys
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest


@pytest.mark.parametrize("dummy", [None])
def test_train_and_export_runs_without_side_effects(monkeypatch, dummy):
    # Avoid filesystem writes
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, parents=True, exist_ok=True: None)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    monkeypatch.setattr("pathlib.Path.stat", lambda self: SimpleNamespace(st_size=1))
    monkeypatch.setattr("builtins.open", mock_open())

    # Stub persistence and copying
    monkeypatch.setattr("numpy.save", lambda *args, **kwargs: None)
    monkeypatch.setattr("joblib.dump", lambda *args, **kwargs: None)
    monkeypatch.setattr("shutil.copytree", lambda *args, **kwargs: None)

    # Stub gameplay to avoid real training and NotFitted errors
    monkeypatch.setattr("Arena.Arena.playHands", lambda self, n: ([], [], [0] * max(1, n)))

    # Ensure a clean import triggers the script once under these stubs
    sys.modules.pop("train_and_export", None)
    module = importlib.import_module("train_and_export")

    # Basic sanity: module was imported and configured a training round count
    assert getattr(module, "TRAINING_ROUNDS", None) is not None
