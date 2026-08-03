"""Testes unitários do carregamento do modelo de churn."""

import joblib
import pytest

from churn_prediction import model_loader


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Isola os testes limpando o cache do modelo entre cada execução."""
    model_loader.load_model.cache_clear()
    yield
    model_loader.load_model.cache_clear()


def test_load_model_reads_existing_artifact(tmp_path):
    model_path = tmp_path / "champion_model.joblib"
    expected_model = {"model": "fake-champion"}
    joblib.dump(expected_model, model_path)

    loaded_model = model_loader.load_model(model_path)

    assert loaded_model == expected_model


def test_load_model_raises_error_when_artifact_does_not_exist(tmp_path):
    missing_model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        model_loader.load_model(missing_model_path)


def test_load_model_reuses_cached_artifact(tmp_path, monkeypatch):
    model_path = tmp_path / "champion_model.joblib"
    model_path.touch()
    expected_model = object()
    load_calls = []

    def fake_joblib_load(received_path):
        load_calls.append(received_path)
        return expected_model

    monkeypatch.setattr(model_loader.joblib, "load", fake_joblib_load)

    first_load = model_loader.load_model(model_path)
    second_load = model_loader.load_model(model_path)

    assert first_load is expected_model
    assert second_load is expected_model
    assert load_calls == [model_path]
