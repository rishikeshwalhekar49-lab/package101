"""Checks (and automatic repairs) for distributed-training framework configs.

Studies of DeepSpeed / Megatron-LM / Colossal-AI bugs find configuration errors
and missing preconditions among the dominant root causes, with most fixes under
10 lines -- a good fit for rule-based repair. Supported inputs: DeepSpeed JSON
configs and Hugging Face ``TrainingArguments`` dictionaries (JSON/YAML-as-JSON).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..types import Evidence, Severity


def _ev(rule: str, msg: str, conf: float, sev: Severity = Severity.WARNING, file: str | None = None, **data: Any) -> Evidence:
    return Evidence(id="", source="config", rule=rule, message=msg, severity=sev, confidence=conf, file=file,
                    category="framework_config", cause="framework_config", fixable=rule in FIXABLE, data=data)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


FIXABLE = {"DS001", "DS002", "DS003", "DS005", "DS007", "DS010", "HF001", "HF003"}


def check_deepspeed_config(cfg: dict[str, Any], world_size: int | None = None, file: str | None = None) -> list[Evidence]:
    out: list[Evidence] = []
    tb = cfg.get("train_batch_size")
    mb = cfg.get("train_micro_batch_size_per_gpu")
    ga = cfg.get("gradient_accumulation_steps", 1)
    if _is_num(tb) and _is_num(mb) and _is_num(ga):
        if world_size and tb != mb * ga * world_size:
            out.append(_ev("DS001", f"train_batch_size={tb} != micro_batch {mb} x grad_accum {ga} x world_size {world_size} = {mb * ga * world_size}",
                           0.95, Severity.ERROR, file, expected=mb * ga * world_size))
        elif not world_size and tb % (mb * ga):
            out.append(_ev("DS001", f"train_batch_size={tb} is not divisible by micro_batch x grad_accum = {mb * ga}", 0.9, Severity.ERROR, file))
    if _is_num(ga) and ga < 1:
        out.append(_ev("DS009", f"gradient_accumulation_steps={ga} must be >= 1", 0.95, Severity.ERROR, file))
    fp16 = bool(cfg.get("fp16", {}).get("enabled")) if isinstance(cfg.get("fp16"), dict) else False
    bf16 = bool(cfg.get("bf16", {}).get("enabled")) if isinstance(cfg.get("bf16"), dict) else False
    if fp16 and bf16:
        out.append(_ev("DS002", "fp16 and bf16 are both enabled", 0.95, Severity.ERROR, file))
    zero = cfg.get("zero_optimization", {}) if isinstance(cfg.get("zero_optimization"), dict) else {}
    stage = zero.get("stage", 0)
    if _is_num(stage) and stage not in (0, 1, 2, 3):
        out.append(_ev("DS004", f"zero_optimization.stage={stage} is not one of 0, 1, 2, 3", 0.95, Severity.ERROR, file))
    if zero.get("offload_param") and _is_num(stage) and stage < 3:
        out.append(_ev("DS003", f"offload_param requires ZeRO stage 3 (stage is {stage}); parameters will not be offloaded", 0.8, Severity.WARNING, file))
    if _is_num(stage) and stage == 3 and zero.get("stage3_gather_16bit_weights_on_model_save") is False:
        out.append(_ev("DS007", "ZeRO-3 with stage3_gather_16bit_weights_on_model_save=false saves partitioned/empty weights "
                       "from save_pretrained()/save_model()", 0.6, Severity.WARNING, file))
    opt = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), dict) else {}
    oparams = opt.get("params", {}) if isinstance(opt.get("params"), dict) else {}
    sched = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler"), dict) else {}
    sparams = sched.get("params", {}) if isinstance(sched.get("params"), dict) else {}
    if sched.get("type") in ("WarmupLR", "WarmupDecayLR", "WarmupCosineLR") and _is_num(sparams.get("warmup_max_lr")) and _is_num(oparams.get("lr")):
        if abs(sparams["warmup_max_lr"] - oparams["lr"]) > 1e-12:
            out.append(_ev("DS005", f"scheduler warmup_max_lr={sparams['warmup_max_lr']} overrides optimizer lr={oparams['lr']}; "
                           "the effective peak LR is not the configured one", 0.6, Severity.WARNING, file))
    if sched.get("type") == "WarmupDecayLR" and _is_num(sparams.get("total_num_steps")) and _is_num(sparams.get("warmup_num_steps")):
        if sparams["total_num_steps"] <= sparams["warmup_num_steps"]:
            out.append(_ev("DS006", f"WarmupDecayLR total_num_steps={sparams['total_num_steps']} <= warmup_num_steps={sparams['warmup_num_steps']}; "
                           "the LR never decays", 0.85, Severity.ERROR, file))
    if str(opt.get("type", "")).lower() == "adam" and _is_num(oparams.get("weight_decay")) and oparams.get("weight_decay", 0) > 0 \
            and oparams.get("adam_w_mode") is False:
        out.append(_ev("DS010", "Adam with weight_decay and adam_w_mode=false applies L2 regularisation, not decoupled AdamW decay",
                       0.55, Severity.WARNING, file))
    autos = [k for k, v in _walk(cfg) if v == "auto"]
    if autos:
        out.append(_ev("DS008", f"'auto' values ({', '.join(autos[:5])}) only resolve under the Hugging Face integration",
                       0.4, Severity.INFO, file))
    return out


def fix_deepspeed_config(cfg: dict[str, Any], world_size: int | None = None) -> tuple[dict[str, Any], list[str]]:
    new = copy.deepcopy(cfg)
    changes: list[str] = []
    for ev in check_deepspeed_config(cfg, world_size):
        if ev.rule == "DS001" and "expected" in ev.data:
            new["train_batch_size"] = ev.data["expected"]
            changes.append(f"train_batch_size -> {ev.data['expected']}")
        elif ev.rule == "DS002":
            new["fp16"]["enabled"] = False
            changes.append("fp16.enabled -> false (keeping bf16)")
        elif ev.rule == "DS003":
            new["zero_optimization"].pop("offload_param", None)
            changes.append("removed zero_optimization.offload_param (needs stage 3)")
        elif ev.rule == "DS005":
            new["scheduler"]["params"]["warmup_max_lr"] = new["optimizer"]["params"]["lr"]
            changes.append(f"scheduler.params.warmup_max_lr -> {new['optimizer']['params']['lr']}")
        elif ev.rule == "DS007":
            new["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
            changes.append("zero_optimization.stage3_gather_16bit_weights_on_model_save -> true")
        elif ev.rule == "DS010":
            new["optimizer"]["params"]["adam_w_mode"] = True
            changes.append("optimizer.params.adam_w_mode -> true")
    return new, changes


def check_hf_training_args(d: dict[str, Any], file: str | None = None) -> list[Evidence]:
    out: list[Evidence] = []
    if d.get("fp16") and d.get("bf16"):
        out.append(_ev("HF001", "fp16 and bf16 are both enabled", 0.95, Severity.ERROR, file))
    ms, ws, wr = d.get("max_steps", -1), d.get("warmup_steps", 0), d.get("warmup_ratio", 0.0)
    if _is_num(ms) and ms > 0 and _is_num(ws) and ws >= ms:
        out.append(_ev("HF002", f"warmup_steps={ws} >= max_steps={ms}; the LR never leaves warm-up", 0.85, Severity.ERROR, file))
    if _is_num(wr) and wr >= 1:
        out.append(_ev("HF002", f"warmup_ratio={wr} >= 1; the LR never leaves warm-up", 0.85, Severity.ERROR, file))
    metric = str(d.get("metric_for_best_model") or "")
    if d.get("load_best_model_at_end") and metric:
        gib = d.get("greater_is_better")
        if "loss" in metric and gib is True:
            out.append(_ev("HF003", f"metric_for_best_model={metric!r} with greater_is_better=true keeps the *worst* checkpoint",
                           0.85, Severity.ERROR, file))
        if "loss" not in metric and gib is False:
            out.append(_ev("HF003", f"metric_for_best_model={metric!r} with greater_is_better=false keeps the *worst* checkpoint",
                           0.6, Severity.WARNING, file))
    ls = d.get("label_smoothing_factor", 0.0)
    if _is_num(ls) and ls > 0.5:
        out.append(_ev("HF004", f"label_smoothing_factor={ls} is implausibly large", 0.6, Severity.WARNING, file))
    lr = d.get("learning_rate")
    if _is_num(lr) and lr > 1e-2:
        out.append(_ev("HF006", f"learning_rate={lr} is very high for fine-tuning", 0.4, Severity.WARNING, file))
    return out


def fix_hf_training_args(d: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    new = dict(d)
    changes = []
    for ev in check_hf_training_args(d):
        if ev.rule == "HF001":
            new["fp16"] = False
            changes.append("fp16 -> false (keeping bf16)")
        elif ev.rule == "HF003":
            new["greater_is_better"] = "loss" not in str(d.get("metric_for_best_model"))
            changes.append(f"greater_is_better -> {new['greater_is_better']}")
    return new, changes


def _walk(d: Any, prefix: str = ""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, d


def detect_kind(d: dict[str, Any]) -> str:
    ds_keys = {"train_batch_size", "train_micro_batch_size_per_gpu", "zero_optimization", "fp16", "bf16", "gradient_accumulation_steps"}
    hf_keys = {"output_dir", "per_device_train_batch_size", "learning_rate", "num_train_epochs", "load_best_model_at_end"}
    if len(ds_keys & set(d)) >= len(hf_keys & set(d)) and ("zero_optimization" in d or "train_micro_batch_size_per_gpu" in d or "train_batch_size" in d):
        return "deepspeed"
    return "hf"


def check_config_file(path: str | Path, world_size: int | None = None) -> tuple[str, list[Evidence]]:
    d = json.loads(Path(path).read_text())
    kind = detect_kind(d)
    if kind == "deepspeed":
        return kind, check_deepspeed_config(d, world_size, str(path))
    return kind, check_hf_training_args(d, str(path))


def fix_config_file(path: str | Path, out: str | Path, world_size: int | None = None) -> list[str]:
    d = json.loads(Path(path).read_text())
    if detect_kind(d) == "deepspeed":
        new, changes = fix_deepspeed_config(d, world_size)
    else:
        new, changes = fix_hf_training_args(d)
    Path(out).write_text(json.dumps(new, indent=2) + "\n")
    return changes
