#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Score definitions and aggregation adapted from VBench (Apache-2.0).
# Adaptations by the UnStep authors. See THIRD_PARTY_NOTICES.md.

import json
import math
import sys
import time
from pathlib import Path

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

NORMALIZE = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

QUALITY = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]


def latest_result(path: Path) -> Path:
    files = sorted(path.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no *_eval_results.json under {path}")
    return files[-1]


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("usage: merge_vbench_scores.py RUN_NAME OUT_JSON OUT_MD DIM_RESULT_DIR...")
    run_name = sys.argv[1]
    out_json = Path(sys.argv[2])
    out_md = Path(sys.argv[3])
    result_dirs = [Path(p) for p in sys.argv[4:]]

    raw = {}
    source_jsons = []
    for result_dir in result_dirs:
        path = latest_result(result_dir)
        source_jsons.append(str(path))
        data = json.loads(path.read_text())
        for dim, value in data.items():
            key = dim.replace("_", " ")
            score = float(value[0])
            if not math.isfinite(score):
                raise SystemExit(f"nonfinite VBench score for {dim} in {path}: {score}")
            raw[key] = score

    expected = set(DIM_WEIGHT)
    missing = sorted(expected - set(raw))
    if missing:
        raise SystemExit(f"missing VBench dimensions: {missing}")

    normalized = {}
    for key in sorted(raw):
        bounds = NORMALIZE[key]
        score = (raw[key] - bounds["Min"]) / (bounds["Max"] - bounds["Min"])
        normalized[key] = score * DIM_WEIGHT[key]

    quality = sum(normalized[k] for k in QUALITY) / sum(DIM_WEIGHT[k] for k in QUALITY)
    semantic = sum(normalized[k] for k in SEMANTIC) / sum(DIM_WEIGHT[k] for k in SEMANTIC)
    total = (4 * quality + semantic) / 5
    norm16 = sum(normalized[k] for k in normalized) / len(normalized)

    out = {
        "run_name": run_name,
        "created_unix": time.time(),
        "dimensions": sorted(raw),
        "source_jsons": source_jsons,
        "raw_dimension_scores": {k: raw[k] for k in sorted(raw)},
        "normalized_dimension_scores_percent": {
            k: normalized[k] * 100.0 for k in sorted(normalized)
        },
        "vbench_quality": quality * 100.0,
        "vbench_semantic": semantic * 100.0,
        "vbench_total": total * 100.0,
        "vbench_historical_weighted_norm16": norm16 * 100.0,
        # Retained for consumers of older reports; this legacy key is misnamed.
        "vbench_unweighted_normalized_16dim_mean": norm16 * 100.0,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")

    lines = [
        f"# {run_name} VBench",
        "",
        f"- Total: {out['vbench_total']:.4f}",
        f"- Quality: {out['vbench_quality']:.4f}",
        f"- Semantic: {out['vbench_semantic']:.4f}",
        f"- Historical weighted Norm16 (auxiliary): {out['vbench_historical_weighted_norm16']:.4f}",
        "",
        "| Dimension | Raw | Normalized % |",
        "|---|---:|---:|",
    ]
    for key in sorted(raw):
        lines.append(f"| {key} | {raw[key]:.6f} | {normalized[key] * 100.0:.4f} |")
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
