"""Silent bugs in *agent-written* training code.

1. ``generate``: ask an LLM coding agent to write training scripts for a set of
   task specs (synthetic data, CPU, reports its metric via trainmedic).
2. ``audit``: run static analysis and the runtime invariants on each script and
   write a labelling sheet (``labels.jsonl``) plus a guide, so humans can
   confirm/deny each signal and tag bug types. Freshly generated code is, by
   construction, not in any model's training data (contamination-resistant).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

from ..api import check_script
from ..llm.backend import LLMBackend

TASK_SPECS: list[dict[str, str]] = [
    {"id": "mlp_tabular", "task": "multi-class classification of synthetic tabular data (20 features, 5 classes) with an MLP using dropout and batch-norm, a train/validation split and per-epoch validation accuracy", "metric": "val_acc", "direction": "max"},
    {"id": "cnn_images", "task": "classification of synthetic 16x16 single-channel images (4 procedurally generated pattern classes) with a small CNN, input normalisation and data augmentation for training only", "metric": "val_acc", "direction": "max"},
    {"id": "char_lm", "task": "a character-level causal transformer language model written from scratch (self-attention with a causal mask) on a synthetic repeating-pattern corpus, reporting validation loss", "metric": "val_loss", "direction": "min"},
    {"id": "grad_accum", "task": "an MLP regressor trained with gradient accumulation over 4 micro-batches, gradient clipping and a cosine learning-rate schedule", "metric": "val_mse", "direction": "min"},
    {"id": "binary_bce", "task": "binary classification with BCEWithLogitsLoss on synthetic data, class imbalance handled with pos_weight", "metric": "val_auc_proxy", "direction": "max"},
    {"id": "seq2seq_rnn", "task": "a GRU sequence model that learns to reverse short integer sequences, with teacher forcing", "metric": "val_token_acc", "direction": "max"},
    {"id": "dpo_toy", "task": "DPO fine-tuning of a small policy network against a frozen reference policy on synthetic preference pairs", "metric": "pref_acc", "direction": "max"},
    {"id": "finetune_head", "task": "fine-tuning only the classification head of a small pretrained-style backbone (randomly initialised then frozen) with a separate learning rate for the head", "metric": "val_acc", "direction": "max"},
]

GEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["filename", "code", "metric", "direction"],
    "properties": {
        "filename": {"type": "string"},
        "code": {"type": "string"},
        "metric": {"type": "string"},
        "direction": {"type": "string", "enum": ["max", "min"]},
    },
}

GEN_SYSTEM = "You are a senior ML engineer. Write complete, runnable, idiomatic PyTorch training scripts."

GEN_PROMPT = """Write a complete PyTorch training script for this task:

{task}

Requirements:
- Generate all data synthetically in the script with torch (no downloads, no files); CPU only; finish in under 60 seconds.
- `import trainmedic as tm`; call `tm.seed_everything()` once at start; wrap the number of epochs/steps as `tm.scaled(N)`.
- At the end call `tm.report({metric}=<final value>)`.
- Guard the entry point with `if __name__ == "__main__":`.
Return JSON with filename, code, metric ("{metric}") and direction ("{direction}")."""

LABELING_GUIDE = """# Labelling guide for agent-generated training scripts

Each line of `labels.jsonl` is one script with the signals trainmedic found.
Fill in, by reading the code (and re-running if needed):

- `label`: `buggy` (at least one silent training bug), `clean`, or `unsure`.
- `bug_types`: list of root-cause ids from `trainmedic taxonomy` (e.g. `missing_eval_mode`,
  `double_softmax`, `grad_accum_unscaled`, `train_eval_leakage`, `future_leakage`).
- `true_positive_evidence`: evidence ids that point at a real bug; the rest are false positives.
- `missed`: free text for bugs no signal pointed at.
- `notes`.

A *silent* bug runs without raising but trains or evaluates incorrectly. Crashes are
recorded separately (`crashed: true`) and are not silent bugs.
"""


def generate(backend: LLMBackend, out_dir: str | Path, *, specs: list[dict[str, str]] | None = None, n: int = 1,
             log=print) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for spec in specs or TASK_SPECS:
        for k in range(n):
            prompt = GEN_PROMPT.format(task=spec["task"], metric=spec["metric"], direction=spec["direction"])
            try:
                data = backend.complete_json(GEN_SYSTEM, prompt, GEN_SCHEMA)
            except Exception as exc:  # keep going; record the failure
                log(f"{spec['id']}#{k}: generation failed: {exc}")
                continue
            name = re.sub(r"[^\w.-]", "_", f"{spec['id']}_{k}.py")
            path = out / name
            path.write_text(data["code"])
            meta = {"file": name, "spec": spec["id"], "metric": data.get("metric", spec["metric"]),
                    "direction": data.get("direction", spec["direction"]), "generator": getattr(backend, "name", "llm"),
                    "created": _dt.date.today().isoformat()}
            made.append(meta)
            log(f"wrote {path}")
    (out / "generated.json").write_text(json.dumps(made, indent=2))
    return made


def audit(out_dir: str | Path, *, max_steps: int = 40, log=print) -> list[dict[str, Any]]:
    out = Path(out_dir)
    meta = {m["file"]: m for m in json.loads((out / "generated.json").read_text())} if (out / "generated.json").exists() else {}
    rows = []
    for f in sorted(out.glob("*.py")):
        res = check_script(str(f), max_steps=max_steps)
        act = res.actionable()
        row = {
            **meta.get(f.name, {"file": f.name}),
            "crashed": not res.ok and res.trace is None,
            "error": res.error,
            "evidence": [e.to_dict() for e in res.evidence],
            "n_actionable": len(act),
            "label": None,
            "bug_types": [],
            "true_positive_evidence": [],
            "missed": "",
            "notes": "",
        }
        rows.append(row)
        log(f"{f.name}: {len(act)} actionable signal(s){' (crashed)' if row['crashed'] else ''}")
    with (out / "labels.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    (out / "LABELING.md").write_text(LABELING_GUIDE)
    return rows


def summarize_labels(out_dir: str | Path) -> dict[str, Any]:
    rows = [json.loads(l) for l in (Path(out_dir) / "labels.jsonl").read_text().splitlines() if l.strip()]
    labelled = [r for r in rows if r.get("label") in ("buggy", "clean")]
    tp = sum(len(r.get("true_positive_evidence", [])) for r in labelled)
    flagged = sum(r.get("n_actionable", 0) for r in labelled)
    bug_types: dict[str, int] = {}
    for r in labelled:
        for b in r.get("bug_types", []):
            bug_types[b] = bug_types.get(b, 0) + 1
    return {
        "scripts": len(rows),
        "labelled": len(labelled),
        "buggy": sum(r["label"] == "buggy" for r in labelled),
        "crashed": sum(bool(r.get("crashed")) for r in rows),
        "signal_precision": tp / flagged if flagged else None,
        "bug_types": dict(sorted(bug_types.items(), key=lambda kv: -kv[1])),
    }
