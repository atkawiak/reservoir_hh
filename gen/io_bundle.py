"""Bundle I/O — write, hash, manifest."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import GeneratorConfig


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bundle(
    out_dir: Path,
    cfg: GeneratorConfig,
    seeds: dict[str, int],
    pop: dict,
    edges: dict,
    neuron_params: dict[str, np.ndarray],
    syn_params: dict,
    base_stats: dict,
    regimes: list[dict],
    poisson_trains: list[np.ndarray],
) -> Path:
    """Write all bundle artifacts to disk.

    Directory structure:
        out_dir/
            config.json
            network/
                population.json
                edges.npz
                neuron_params.npz
                synapse_params.npz
                base_stats.json
            regimes/
                regimes.json
            poisson/
                trains_3s.npz
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── config.json ──
    cfg_dict = _config_to_dict(cfg)
    cfg_dict["seeds"] = seeds
    _write_json(out_dir / "config.json", cfg_dict)

    # ── network/ ──
    net_dir = out_dir / "network"
    net_dir.mkdir(exist_ok=True)

    _write_json(net_dir / "population.json", {
        "N_E": int(pop["N_E"]),
        "N_I": int(pop["N_I"]),
        "E_idx": pop["E_idx"].tolist(),
        "I_idx": pop["I_idx"].tolist(),
    })

    np.savez_compressed(
        net_dir / "edges.npz",
        pre=edges["pre"], post=edges["post"], block_id=edges["block_id"],
    )

    np.savez_compressed(net_dir / "neuron_params.npz", **neuron_params)

    np.savez_compressed(
        net_dir / "synapse_params.npz",
        w=syn_params["w"],
        U=syn_params["U"],
        tau_d_ms=syn_params["tau_d_ms"],
        tau_f_ms=syn_params["tau_f_ms"],
        delay_ms=syn_params["delay_ms"],
    )

    _write_json(net_dir / "base_stats.json", base_stats)

    # ── regimes/ ──
    reg_dir = out_dir / "regimes"
    reg_dir.mkdir(exist_ok=True)
    _write_json(reg_dir / "regimes.json", regimes)

    # ── poisson/ ──
    poi_dir = out_dir / "poisson"
    poi_dir.mkdir(exist_ok=True)
    poi_data = {"n_channels": len(poisson_trains),
                "T_s": cfg.poisson_T_s,
                "rate_hz": cfg.poisson_rate_hz,
                "seed": seeds["poisson"]}
    for ch, t in enumerate(poisson_trains):
        poi_data[f"ch{ch}"] = t
    np.savez_compressed(poi_dir / "trains_3s.npz", **poi_data)

    return out_dir


def write_manifest(bundle_dir: Path) -> Path:
    """Create manifest.json with SHA-256 hashes for all data files."""
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {},
    }

    for ext in ("*.json", "*.npz"):
        for p in sorted(bundle_dir.rglob(ext)):
            if p.name == "manifest.json":
                continue
            rel = str(p.relative_to(bundle_dir))
            manifest["files"][rel] = sha256_file(p)

    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def verify_manifest(bundle_dir: Path) -> list[str]:
    """Verify bundle integrity against manifest. Returns list of errors."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json not found"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    for rel, expected in manifest.get("files", {}).items():
        full = bundle_dir / rel
        if not full.exists():
            errors.append(f"missing: {rel}")
            continue
        actual = sha256_file(full)
        if actual != expected:
            errors.append(f"hash mismatch: {rel}")

    return errors


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _config_to_dict(cfg: GeneratorConfig) -> dict:
    """Convert GeneratorConfig to a JSON-safe dict."""
    d = {}
    for k, v in cfg.__dict__.items():
        if k == "neuron_hetero":
            d[k] = {name: {"base": hp.base, "sigma": hp.sigma,
                           "clamp_lo": hp.clamp_lo, "clamp_hi": hp.clamp_hi}
                    for name, hp in v.items()}
        elif k == "stp_params":
            d[k] = {name: {"U": sp.U, "tau_d_ms": sp.tau_d_ms,
                           "tau_f_ms": sp.tau_f_ms, "sigma": sp.sigma}
                    for name, sp in v.items()}
        else:
            d[k] = v
    return d
