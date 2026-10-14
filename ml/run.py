"""Single entry point for every experiment, local CPU or cloud GPU.

Examples (from the repository root):
    python -m ml.run centralized
    python -m ml.run centralized --device auto                     # GPU if available
    python -m ml.run federated --config configs/federated_fast.yaml --set federated.rounds=4 --set output_tag=demo
    python -m ml.run dp --set privacy.noise_multipliers=[1.0]
    python -m ml.run jax
    python -m ml.run ray-benchmark

`--set a.b=c` overrides any YAML key (values parsed as YAML). The merged config is written
next to the results so every run is reproducible from its own files. Nothing here reads or
stores credentials; cloud authentication is handled by the provider's CLI/environment.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "centralized": "configs/centralized.yaml",
    "federated": "configs/federated_fast.yaml",
    "dp": "configs/federated_dp.yaml",
    "jax": "configs/jax_cnn.yaml",
    "ray-benchmark": "configs/ray_benchmark.yaml",
}


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--config", type=Path, help="YAML config (default depends on the experiment)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"],
                        help="device for the PyTorch MLP (centralized / jax experiments); federated runs are CPU")
    args = parser.parse_args(argv)

    config_path = args.config or ROOT / DEFAULT_CONFIGS[args.experiment]
    cfg = apply_overrides(yaml.safe_load(config_path.read_text(encoding="utf-8")), args.set)
    if args.device:
        # centralized/jax: the PyTorch MLP config lives under `mlp`; federated/dp: only the centralized
        # reference MLP can use a GPU (client training is CPU, see docs/LIGHTNING_AI.md).
        key = "reference_training" if args.experiment in ("federated", "dp") else "mlp"
        cfg.setdefault(key, {})["device"] = args.device

    # Write the merged config to a temp file so each experiment's own CLI stays unchanged.
    tmp = Path(tempfile.mkdtemp()) / f"{config_path.stem}_merged.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"experiment={args.experiment} config={config_path} overrides={args.set} device={args.device or 'config'}")

    if args.experiment == "centralized":
        from ml.train_centralized import main as run
    elif args.experiment == "federated":
        from ml.federated.experiment import main as run
    elif args.experiment == "dp":
        from ml.federated.dp_experiment import main as run
    elif args.experiment == "jax":
        from ml.jax_cnn.experiment import main as run
    else:
        from ml.federated.ray_benchmark import main as run
    run(tmp)


if __name__ == "__main__":
    main()
