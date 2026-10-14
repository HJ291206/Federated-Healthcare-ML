import pytest

from ml.run import DEFAULT_CONFIGS, ROOT, apply_overrides


def test_default_configs_exist():
    for rel in DEFAULT_CONFIGS.values():
        assert (ROOT / rel).is_file(), rel


def test_overrides_parse_yaml_values_and_nested_keys():
    cfg = {"seeds": [0, 1], "federated": {"rounds": 8, "lr": 0.1}}
    out = apply_overrides(cfg, ["federated.rounds=4", "seeds=[0]", "mlp.device=cuda", "federated.lr=0.05"])
    assert out["federated"] == {"rounds": 4, "lr": 0.05}
    assert out["seeds"] == [0] and out["mlp"] == {"device": "cuda"}
    assert cfg["federated"]["rounds"] == 8  # input not mutated


def test_malformed_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["rounds"])
