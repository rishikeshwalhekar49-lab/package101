"""Root-cause taxonomy for silent DL training faults.

Categories follow the DEFault/DEFault++ hierarchy (detect -> category -> root
cause) extended with attention-specific causes from "Why Attention Fails"
(ICSE'26), RL post-training causes and framework/compiler/distributed causes so
that diagnosis can also *attribute* a symptom to user code vs. framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cause:
    id: str
    category: str
    title: str
    fix: str
    attribution: str = "user_code"
    related: tuple[str, ...] = field(default_factory=tuple)


_C = [
    # optimization
    Cause("missing_zero_grad", "optimization", "Gradients are never cleared between steps",
          "Call optimizer.zero_grad() once per optimizer step (after step(), or before the first backward of a step)."),
    Cause("grads_cleared_before_step", "optimization", "Gradients are cleared between backward() and step()",
          "Move zero_grad() so it runs before backward() (or right after step()), never between backward() and step()."),
    Cause("step_without_backward", "optimization", "Optimizer steps without fresh gradients",
          "Ensure every optimizer.step() is preceded by a backward() of the current batch."),
    Cause("params_not_updated", "optimization", "Optimizer step leaves parameters unchanged",
          "Check the learning rate (non-zero), the parameters passed to the optimizer, and requires_grad flags."),
    Cause("optimizer_coverage", "optimization", "Trainable parameters are missing from the optimizer",
          "Pass model.parameters() (or every trainable sub-module) to the optimizer, or freeze the rest with requires_grad_(False)."),
    Cause("grad_accum_unscaled", "optimization", "Gradient accumulation without loss scaling",
          "Divide the loss by the number of accumulation steps before backward()."),
    Cause("grad_clip_misplaced", "optimization", "Gradient clipping has no effect",
          "Clip after backward() and before optimizer.step()."),
    Cause("lr_schedule_misuse", "optimization", "Learning-rate schedule misapplied",
          "Step epoch-based schedulers once per epoch and per-iteration schedulers once per optimizer step, after optimizer.step()."),
    Cause("lr_scale", "optimization", "Learning rate is badly scaled",
          "Lower (or raise) the learning rate so that ||dW||/||W|| per step is roughly 1e-4..1e-2.", related=("loss_spike",)),
    Cause("no_learning", "optimization", "Training makes far less progress than healthy runs",
          "Look for disconnected graphs, frozen parameters, wrong targets or a tiny learning rate."),
    # loss
    Cause("double_softmax", "loss", "Softmax applied before cross-entropy",
          "Feed raw logits to CrossEntropyLoss / F.cross_entropy; remove the explicit softmax."),
    Cause("double_sigmoid", "loss", "Sigmoid applied before BCEWithLogitsLoss",
          "Feed raw logits to BCEWithLogitsLoss; remove the explicit sigmoid."),
    Cause("nll_without_log_softmax", "loss", "NLL loss on raw scores",
          "Apply log_softmax before NLLLoss, or use CrossEntropyLoss on logits."),
    Cause("lm_target_not_shifted", "attention", "Language-model targets are not shifted",
          "Predict token t+1 from tokens <= t: logits[:, :-1] against targets[:, 1:].", related=("target_leakage",)),
    Cause("target_leakage", "data", "Targets leak into the inputs",
          "Make sure the model cannot see the label: shift LM targets, mask future tokens, separate features from labels.",
          related=("lm_target_not_shifted", "future_leakage")),
    # model mode
    Cause("missing_eval_mode", "model_mode", "Evaluation runs in training mode",
          "Call model.eval() before evaluation (and model.train() before training resumes)."),
    Cause("missing_train_mode", "model_mode", "Training runs in eval mode",
          "Call model.train() at the start of every training epoch / after each evaluation."),
    Cause("dropout_in_eval", "model_mode", "Dropout stays active at evaluation time",
          "Pass training=self.training to functional dropout, or use nn.Dropout modules."),
    # graph
    Cause("detached_graph", "graph", "Prediction is detached before the loss",
          "Remove .detach()/.data/no_grad() on the path from model output to the loss."),
    Cause("disconnected_params", "graph", "Some parameters never receive gradients",
          "Check for .detach(), torch.no_grad(), unused layers, or layers created but not called in forward().",
          related=("detached_graph",)),
    Cause("graph_retention", "resource", "Autograd graphs kept alive by tensor accumulation",
          "Accumulate loss.item() (or loss.detach()) instead of the loss tensor."),
    Cause("hooks_bypassed", "model_mode", "Module hooks bypassed by calling forward() directly",
          "Call model(x) instead of model.forward(x)."),
    # data
    Cause("no_shuffle", "data", "Training data is not shuffled",
          "Use DataLoader(..., shuffle=True) (or a shuffling sampler) for the training set."),
    Cause("train_eval_leakage", "data", "Evaluation data overlaps training data",
          "Build evaluation loaders from a held-out split that is disjoint from training data."),
    Cause("preprocessing_mismatch", "data", "Training and evaluation preprocessing differ",
          "Apply the same normalisation/preprocessing to evaluation inputs as to training inputs."),
    Cause("eval_augmentation", "data", "Random augmentation at evaluation time",
          "Use deterministic transforms for validation/test data."),
    # attention / transformer
    Cause("attention_scale_missing", "attention", "Attention scores are not scaled",
          "Multiply Q @ K^T by 1/sqrt(head_dim) (or use F.scaled_dot_product_attention)."),
    Cause("attention_softmax_dim", "attention", "Attention softmax over the wrong axis",
          "Normalise attention weights over the key axis: softmax(scores, dim=-1)."),
    Cause("causal_mask_inverted", "attention", "Causal mask is inverted",
          "With a tril mask of ones, mask positions where mask == 0 (the future), not where mask == 1.",
          related=("future_leakage",)),
    Cause("future_leakage", "attention", "Future tokens influence past positions",
          "Apply a causal mask (masked_fill(mask == 0, -inf) with a tril mask, or is_causal=True).",
          related=("causal_mask_inverted", "lm_target_not_shifted")),
    Cause("batch_mixing", "model_structure", "Computation mixes samples across the batch",
          "Fix the view/reshape/permute so the batch dimension stays first and is never merged with other dims."),
    # numerics
    Cause("nonfinite_loss", "numerics", "Loss becomes NaN/Inf",
          "Lower the LR, add gradient clipping, check log/div of zero, or use loss scaling for fp16.",
          attribution="hardware_or_numerics", related=("lr_scale", "loss_spike")),
    Cause("nonfinite_grad", "numerics", "Non-finite gradients", "Clip gradients, lower the LR, check custom autograd code.",
          attribution="hardware_or_numerics"),
    Cause("optimizer_state_corrupt", "numerics", "Optimizer state contains NaN/Inf",
          "Restore optimizer state from a checkpoint and skip the offending batches.", attribution="hardware_or_numerics"),
    Cause("loss_spike", "numerics", "Loss spikes", "Use gradient clipping or rollback-and-skip (Guardian).",
          attribution="hardware_or_numerics", related=("lr_scale",)),
    Cause("init_scale", "numerics", "Badly scaled initialisation", "Initialise the final layer with small weights."),
    Cause("grad_scale", "numerics", "Gradient norm outside the healthy range",
          "Check loss scaling, reduction (sum vs mean) and accumulation scaling."),
    Cause("fp16_no_loss_scaling", "numerics", "float16 training without loss scaling", "Use torch.amp.GradScaler or bfloat16."),
    # RL post-training
    Cause("kl_sign", "rl_post_training", "KL penalty has the wrong sign",
          "Add beta * KL to the loss (subtract it from the reward)."),
    Cause("reference_model_trainable", "rl_post_training", "Reference policy is not frozen",
          "Call ref_model.eval() and ref_model.requires_grad_(False); never pass its parameters to the optimizer."),
    Cause("ratio_not_exp", "rl_post_training", "Policy ratio not exponentiated", "ratio = exp(new_logp - old_logp)."),
    Cause("preference_swap", "rl_post_training", "Chosen and rejected responses are swapped",
          "Compute log-ratios as logp(chosen) - logp(rejected)."),
    # frameworks
    Cause("prng_key_reuse", "framework_jax", "JAX PRNG key reused", "Split the key before each use: key, sub = jax.random.split(key)."),
    Cause("from_logits_mismatch", "framework_keras", "Keras from_logits mismatch",
          "Use from_logits=True with a linear output layer, or from_logits=False with a softmax output."),
    Cause("rank_divergence", "distributed", "Data-parallel replicas diverged",
          "Make sure gradients are all-reduced on every rank (no rank-specific code paths), and resync from rank 0.",
          attribution="distributed"),
    Cause("compiler_mismatch", "compiler", "torch.compile output differs from eager",
          "Run the culprit module eagerly (torch.compiler.disable) and report the miscompilation upstream.",
          attribution="compiler"),
    Cause("framework_config", "framework_config", "Distributed-framework configuration error",
          "Fix the reported configuration key(s); see the config check for the exact rule.", attribution="framework_config"),
    Cause("unseeded", "reproducibility", "No random seed", "Seed torch/numpy/random once at start-up."),
    Cause("crash", "crash", "The run crashed", "Fix the exception shown in the trace."),
    Cause("syntax_error", "crash", "Syntax error", "Fix the syntax error."),
    Cause("unknown", "unknown", "Unclassified anomaly", "Inspect the evidence."),
]

CAUSES: dict[str, Cause] = {c.id: c for c in _C}


def get(cause_id: str | None) -> Cause:
    if cause_id and cause_id in CAUSES:
        return CAUSES[cause_id]
    if cause_id and cause_id.startswith("ds_"):
        return CAUSES["framework_config"]
    return CAUSES["unknown"]


def related(a: str, b: str) -> bool:
    if a == b:
        return True
    return b in get(a).related or a in get(b).related


def catalogue_text() -> str:
    """Compact listing used in LLM prompts."""
    return "\n".join(f"- {c.id} [{c.category}]: {c.title}" for c in _C)
