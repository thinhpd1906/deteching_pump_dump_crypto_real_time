"""Device selection (task A1): cuda > mps > cpu, explicit override, bad values."""
import pytest
import torch

from src.training.common import get_device, make_generator


def test_auto_picks_best_available():
    dev = get_device({"train": {"device": "auto"}})
    if torch.cuda.is_available():
        assert dev == "cuda"
    elif torch.backends.mps.is_available():
        assert dev == "mps"
    else:
        assert dev == "cpu"


def test_missing_key_defaults_to_auto():
    assert get_device({"train": {}}) == get_device({"train": {"device": "auto"}})
    assert get_device(None) == get_device({"train": {"device": "auto"}})


def test_force_cpu():
    assert get_device({"train": {"device": "cpu"}}) == "cpu"


def test_invalid_value_raises():
    with pytest.raises(ValueError):
        get_device({"train": {"device": "tpu"}})


def test_unavailable_device_raises():
    if torch.cuda.is_available():
        pytest.skip("cuda present")
    with pytest.raises(RuntimeError):
        get_device({"train": {"device": "cuda"}})


def test_generator_is_seeded_and_usable():
    dev = get_device(None)
    a = torch.randperm(50, device=make_generator(dev, 7).device,
                       generator=make_generator(dev, 7))
    g = make_generator(dev, 7)
    b = torch.randperm(50, device=g.device, generator=g)
    assert torch.equal(a.cpu(), b.cpu())
