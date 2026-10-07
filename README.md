# trainmedic

**Detect, diagnose, repair and statistically validate silent bugs in deep-learning training code.**

Silent training bugs run without raising an error. The model still trains, just worse, or the evaluation lies. Examples include a missing `zero_grad()`, softmax applied before `CrossEntropyLoss`, evaluating without `model.eval()`, gradient accumulation without loss scaling, an inverted causal mask, unshifted language-model targets, a KL penalty with the wrong sign, or train/validation leakage.

`trainmedic` closes the whole loop for user-written PyTorch training code:

```
 static lint ─┐
              ├─► evidence ─► diagnosis ─► abstain? ─► candidate patches ─► invariant pre-filter
 runtime  ────┘   (S/V/P/C/K ids)  (rules + LLM,           (templates, LLM,     (violated invariant must
 invariants                         attribution)            memory)             hold, no new violations)
                                                                                        │
           minimal patch ◄── delta-debugging ◄── accepted ◄── budgeted multi-seed acceptance
                             (noisy oracle)                   (successive halving + held-out seeds)
```

It is built to **do no harm**. Every stage can abstain:

- Hygiene-only signals never justify an edit.
- LLM claims must cite evidence.
- Patches with collateral risky edits are rejected, such as a changed LR, removed dropout or an altered loss reduction.
- A patch is accepted only when a held-out multi-seed test shows a real improvement.

## Install

```bash
pip install trainmedic            # core (needs PyTorch >= 2.1)
pip install "trainmedic[llm]"     # + Claude-based diagnosis/repair
pip install "trainmedic[all]"     # + SciPy for Welch t-tests
```

From a checkout: `pip install -e ".[dev]"` and run `pytest`.

## 60-second tour

```bash
trainmedic demo                                   # inject a bug into a bundled pipeline and repair it end-to-end
trainmedic lint train.py                          # static rules (no execution)
trainmedic check train.py --max-steps 40          # lint + traced run + training invariants + probes
trainmedic diagnose train.py --llm anthropic      # ranked root causes with attribution
trainmedic repair train.py --metric val_acc       # full loop; prints a report and the validated, minimised diff
trainmedic repair train.py --metric val_acc --out fix.diff --report report.md
```

Arguments after `--` go to your script: `trainmedic check train.py -- --epochs 2 --data ./d`.

To support multi-seed validation and budgets, add these lines to your script (all are no-ops outside trainmedic):

```python
import trainmedic as tm
tm.seed_everything()                 # honours the run seed (fixed manual_seed(42) calls are also re-seeded)
for epoch in range(tm.scaled(20)):   # successive halving runs cheap rungs with a fraction of the budget
    ...
tm.report(val_acc=acc)               # the metric used for statistical acceptance
```

Without `tm.report`, use `--metric train_loss_final --direction min`. The tracer always reports this metric. Without `--metric`, a fix is accepted on invariant evidence alone and labelled `unvalidated_fix`.

### From Python

```python
import trainmedic as tm

with tm.watch(model, optimizer, max_steps=100) as w:   # in-process tracing of your own loop
    train()
print(w.evidence)

res = tm.check_script("train.py", max_steps=40)        # subprocess: lint + trace + invariants
from trainmedic.repair.agent import repair
report = repair("train.py", metric="val_acc", mode="rules")
print(report.to_markdown()); print(report.diff())
```

## What it detects

**Static rules** (`trainmedic rules`; 32 rules, 27 with template fixes):

| area | rules |
|---|---|
| gradient lifecycle | missing `zero_grad` (TM001); `zero_grad` between backward and step (TM002); misplaced clipping (TM003); unscaled gradient accumulation (TM011) |
| LR schedules | scheduler before optimizer (TM004); epoch scheduler stepped per batch (TM005); scheduler never stepped (TM040) |
| train/eval mode | missing `eval()` (TM006); missing `train()` (TM007); `F.dropout` without `training=` (TM014); direct `.forward()` (TM029) |
| loss wiring | softmax→CE (TM008); sigmoid→BCEWithLogits (TM009); NLL on raw scores (TM010); detached prediction (TM015) |
| data | unshuffled training loader (TM012); eval on training data (TM013); eval-time augmentation (TM032); normalisation mismatch (TM033) |
| attention / LMs | missing 1/√d scaling (TM017); softmax over wrong axis (TM018); inverted causal mask (TM019); unshifted LM targets (TM020) |
| RL post-training | KL penalty sign (TM023); trainable reference model (TM024); PPO ratio without `exp` (TM025) |
| other frameworks | JAX PRNG key reuse (TM022); Keras `from_logits` mismatch (TM021); fp16 without GradScaler (TM031) |

Suppress a finding with `# trainmedic: ignore[TM012]` on that line.

**Runtime invariants** (29 invariants; TrainCheck-style tracing through global PyTorch hooks, so scripts need no changes):

- **Gradient lifecycle:** gradients are cleared between steps; gradients are present at the step; there is a backward pass before each step; the step changes parameters; every optimized parameter receives a gradient; the optimizer covers all trainable parameters; gradient accumulation is scaled.
- **Model mode:** the model is in train mode for backpropagated forwards and in eval mode under `no_grad`.
- **Loss inputs:** cross-entropy gets logits; NLL gets log-probabilities; BCE-with-logits gets logits.
- **Numerics:** loss, gradients and optimizer state are finite; the update ratio is sane; there are no loss spikes; the initial loss is plausible.
- **Data:** the loss does not collapse suspiciously fast; training batches are shuffled; train and eval sets are disjoint; eval inputs have the same distribution as training inputs.
- **Distributed:** parameters are identical across data-parallel ranks.
- **Learned from references:** scheduler cadence, gradient-norm range and loss progress.

Three side-effect-free **probes** run on each model's first training forward:

- **eval-mode determinism:** catches `F.dropout` that stays active at eval.
- **batch independence:** catches a reshape that mixes samples across the batch.
- **causality:** catches a missing or inverted attention mask.

Also exposed are `overfit_one_batch`, `initial_loss_check` and `kv_cache_consistency` in `trainmedic.probes`.

**Learn invariants from healthy runs** (removes false alarms on unusual but correct pipelines):

```bash
trainmedic trace good_train.py -o ref1.jsonl --seed 0
trainmedic trace good_train.py -o ref2.jsonl --seed 1
trainmedic infer ref1.jsonl ref2.jsonl -o invariants.json
trainmedic check new_train.py --invariants invariants.json
```

**Differential debugging against a known-good version.** Run both versions with the same seed. `trace-diff` aligns the two traces step by step and reports the first signal that diverges: train/eval mode, LR, loss, gradient norm, update size, loss-input statistics, backward or `zero_grad` cadence.

```bash
trainmedic trace old_train.py -o ref.jsonl && trainmedic trace new_train.py -o new.jsonl
trainmedic trace-diff ref.jsonl new.jsonl
```

**Beyond user code:**

- `trainmedic compile-diff file.py:factory` runs a model eagerly and under `torch.compile`, compares outputs and gradients, and localises a miscompiled sub-module. A mismatch is attributed to the compiler, not your code.
- `trainmedic config ds_config.json --world-size 8 --fix fixed.json` checks and repairs DeepSpeed and Hugging Face `TrainingArguments` configs. It catches batch-size identity errors, fp16+bf16, `offload_param` below ZeRO-3, warm-up overriding the LR, a "best" checkpoint picked as the worst one, and more.

## Diagnosis, repair and abstention

- **Diagnosis** fuses all evidence by root cause. Corroboration across sources, for example a static finding plus a runtime violation, raises confidence. It also **attributes** the problem to user code, framework config, compiler, distributed setup or numerics (`trainmedic taxonomy`).
- **Candidates** come from template fixers, an LLM (optional) and the experience memory (optional, `--memory`).
- **Modes:**
  - `--mode rules`: deterministic and offline.
  - `--mode llm` or `--mode hybrid`: evidence-grounded LLM with an explicit abstain option.
  - `--mode naive-llm`: a deliberately naive baseline used to *measure* damage.
- **Abstention** happens at three gates:
  1. No actionable evidence.
  2. An ungrounded patch, or a patch with collateral risky edits.
  3. No significant improvement on held-out seeds.

### LLM backends

| `--llm` | backend |
|---|---|
| `anthropic` | Claude via the official SDK. The default model is `claude-opus-5-5`, with adaptive thinking, JSON-schema-constrained output, a cached system prompt, and server-side refusal fallback (`fallbacks="default"`) enabled. |
| `anthropic:claude-sonnet-5-5` | a different Claude model |
| `cmd:python my_llm.py` | any local or self-hosted model behind a command (JSON on stdin and stdout; see `examples/llm_command_backend.py`) |

Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile. `TRAINMEDIC_LLM`, `TRAINMEDIC_LLM_MODEL` and `TRAINMEDIC_LLM_EFFORT` set the defaults.

## Statistical validation and minimisation

`SequentialValidator` / `trainmedic validate` accepts candidate patches in three steps:

1. **Successive halving.** Cheap rungs with a few seeds and a partial budget (via `tm.scaled`) rank candidates by their *paired* improvement over the baseline. Each rung keeps the top 1/η and drops futile candidates early.
2. **Held-out confirmation.** Finalists are re-run on fresh seeds at full budget.
3. **Final test.** An exact one-sided sign-flip test at α/#finalists.

Selection and confirmation use disjoint seeds, so **P(accept a non-improving patch) ≤ α**. In simulation the false-accept rate was 4.3% at α=0.05, with 96% power to accept a candidate with a +0.05 true effect. Each report shows the compute used against fixed-k full-budget evaluation.

`trainmedic minimize` (run automatically after acceptance) applies delta debugging over the patch's edits:

- The oracle is three-valued: invariants first, then a multi-seed majority vote. Inconclusive results count as "does not repair".
- It searches the complement for an independent alternative fix.
- It classifies the minimal fix as `code`, `hyperparameter` or `mixed`.

## Online remediation: Guardian

The Guardian responds to triggers during training:

| trigger | likely cause | action |
|---|---|---|
| non-finite gradient | transient overflow | skip the step |
| loss spike or NaN loss | bad batch | roll back to the last snapshot and skip that batch |
| repeated spikes | optimisation instability | roll back and back off the learning rate |
| non-finite optimizer state | corrupted state | restore the optimizer state |
| replica divergence | missed all-reduce or silent data corruption | resync the replicas |
| model in eval mode, or parameters not updating | a code bug | alert (rolling back cannot fix code) |

`SimulatedDataParallel` lets you inject distributed faults on a laptop, for example a rank that skips the all-reduce, corrupted parameters or NaN gradients. See `examples/watch_and_guard.py`.

```python
from trainmedic.remediate import Guardian, run_guarded
g = Guardian(model, optimizer, scheduler, snapshot_every=50, adaptive_clip=True)
# in your loop: loss.backward(); if g.pre_step(step, batch_id): optimizer.step(); d = g.post_step(step, loss, batch_id)
```

## silent_bug_agent: runtime hooks for the Runtime Monitor Agent

`silent_bug_agent.runtime.hooks` is the perception layer for an agent that watches training live. It attaches hooks to an existing model and optimizer, with no change to the model's code, and publishes one `TrainingState` snapshot per monitored step:

| captured | how |
|---|---|
| activation distributions per layer | forward hooks: mean, std, min, max, mean \|x\|, fraction zero, fraction saturated, norm, NaN/Inf count, optional histogram |
| gradients per layer | hooks on each layer's output tensor: norm, mean \|g\|, max \|g\|, NaN/Inf count |
| gradient norms, weight norms, weight deltas per parameter | optimizer step pre/post hooks |
| loss dynamics | `backward()` wrapper; EMA z-score spike tracking |

Each snapshot also carries the learning rates, global norms, the update ratio and a list of `anomalies`:

- `nonfinite_loss`, `nonfinite_activation`, `nonfinite_gradient`, `nonfinite_activation_grad`
- `dead_units`, `saturated_units`
- `loss_spike`, `grad_norm_spike`
- `update_too_large`, `no_update`
- `eval_mode_training`

```python
from silent_bug_agent.runtime import RuntimeHooks, HookConfig

hooks = RuntimeHooks(model, optimizer, HookConfig(every_n_steps=10)).attach()
# ... the training loop stays exactly as it is ...
state = hooks.latest                    # newest TrainingState
for state in hooks.stream:              # or consume from an agent thread (blocks until the next snapshot)
    print(state.summary(), [a.code for a in state.anomalies])
hooks.detach()                          # removes every hook
```

How the overhead stays low:

- Statistics are computed on the tensor's own device from views. Activations are never copied or moved.
- Tensors larger than 65,536 elements are sampled with a strided view. One full reduction still catches any NaN/Inf.
- Hooks never synchronise with the GPU. Each monitored step makes one small asynchronous device-to-host copy into reused pinned buffers.
- The only extra copy is a strided sample of each weight tensor, taken before `optimizer.step()` into a buffer that is allocated once.
- `SnapshotStream` is a bounded ring buffer that never blocks training. If a consumer falls behind, the oldest snapshots are dropped and counted.

Without an optimizer hook (manual updates), call `hooks.end_step()` after each step.

**Validation:**

```bash
silent-bug-agent-validate --steps 20          # or: python -m silent_bug_agent.runtime.validate --device cuda
```

It runs seven architectures: MLP, ResNet with in-place ReLU, LSTM, GRU, Transformer encoder, GPT-style causal decoder with `nn.MultiheadAttention`, and a tanh/sigmoid autoencoder. For each one it checks that:

- training is **bit-identical** with and without hooks;
- loss, gradient norm and weight delta match an independent measurement every step;
- every selected layer reports activation and gradient statistics;
- `detach()` leaves no hook behind.

It then injects six faults (NaN input, dead ReLUs, saturated tanh, ×1e4 learning rate, training in eval mode, a loss spike) and checks that each raises its anomaly. It also checks that a clean run raises none, that an agent thread receives every snapshot, and that a `torch.compile`d model still produces snapshots. All checks pass on CPU. CUDA is supported but has not been run yet.

Measured overhead per training step on CPU (4 threads):

| model | monitor every step | every 10th step |
|---|---|---|
| CNN, 7 conv layers × 64 channels, 32×32, batch 64 (440 ms/step) | 1.12× | 1.00× |
| MLP 784-2048-2048-10, batch 256 (11 ms/step) | 1.85× | 1.00× |
| the validation's toy models (sub-millisecond steps) | 1.3–5× | — |

For long runs use `every_n_steps` of 10 or more.

## Benchmark: do no harm

```bash
trainmedic bench list                                   # bundled pipelines + 20 mutation operators
trainmedic bench build suite/ --cutoff 2026-01-01       # clean twins + buggy twins with ground-truth patches
trainmedic bench verify suite/ --seeds 4                # label mutants: silent_failure / benign / crash
trainmedic bench run suite/ --agent rules --out results/          # repair rate, damage rate, abstention, patch size
trainmedic bench report results/                                   # re-print the summary table
trainmedic bench run suite/ --agent naive-llm --llm anthropic     # measure LLM damage on bug-free code
trainmedic bench mutate my_train.py -o mutants/         # buggy twins of *your own* fresh code (contamination-free)
trainmedic bench agentgen gen/ --llm anthropic && trainmedic bench audit gen/   # audit agent-written training code
```

The bundled pipelines are CPU-only, use synthetic data and take 5–10 s each:

- MLP classifier
- CNN with normalisation and batch-norm
- tiny causal transformer LM with hand-written attention, accumulation and a warm-up/cosine schedule
- DPO-style preference optimisation with a frozen reference and a KL penalty

The agents are `rules`, `llm`, `hybrid`, `naive-llm`, plus `oracle` (applies the ground truth) and `abstain`. These last two are sanity bounds. Results are split by category and by contamination split.

### Results on the bundled suite (rule-based agent, no LLM)

| metric | value |
|---|---|
| silent-failure bugs detected | 92% (23/25) |
| bugs repaired (validated against the clean twin) | 76% (19/25) |
| false alarms on clean pipelines | 0/4 |
| damage on clean pipelines + benign mutants | 0/18 |
| mean patch size | 1.7 changed lines |

Verification classified 14 of the 39 mutants as benign: they do not move the metric. The remaining misses need the LLM modes or a reference run. Details and per-instance results are in [docs/benchmark_results.md](docs/benchmark_results.md).

## How this maps to the 2025–2026 literature

| idea | where |
|---|---|
| (a) invariant detection → LLM diagnosis → verified repair | `repair/agent.py` (the violated invariant serves as diagnosis context, pre-filter and minimisation oracle) |
| (b) abstention and damage rate on bug-free pipelines | `repair/abstention.py`, `bench/evaluate.py` |
| (c) patch minimisation under stochastic training | `repair/minimize.py` |
| (d) silent bugs in agent-generated training code | `bench/agentgen.py` |
| (e) invariants under torch.compile; eager-vs-compiled differential checks | the tracer skips hooks while compiling; `detect/compile_diff.py` |
| (f) broader, cheaper transformer diagnosis | attention rules TM017–TM020, causality/KV-cache probes |
| (g) invariant-triggered, root-cause-conditioned rollback | `remediate/guardian.py`, `remediate/simddp.py` |
| (h) compute-budgeted statistical patch validation | `validate/halving.py` |
| user-code vs framework attribution | `diagnose/engine.py`, `compile_diff`, `config_check` |
| differential debugging against a reference (training-time Ekka) | `detect/trace_diff.py`, `trainmedic trace-diff` |
| distributed-framework config repair | `detect/config_check.py` |
| contamination-aware benchmark split | `bench/suite.py` (`--cutoff`, `source`, `created`) |
| experience memory without over-application | `memory.py` (precision gating; stricter after recorded damage) |
| JAX/Keras and RL post-training code | TM021, TM022 and TM023–TM025 |

## Limitations

- PyTorch is the primary target. JAX and Keras support is static-only, with a few rules.
- The tracer uses Python-level hooks. It sees the boundary of `torch.compile`d modules but not inside compiled graphs. Use `compile-diff` for that.
- Probes run extra forward passes on the first training batch. Disable them with `--no-probes` for models whose forward has side effects (for example, caches).
- Static rules are heuristics with calibrated confidences, and some are intentionally conservative. Learn invariants from a healthy reference run to adapt them to your pipeline.
- Statistical validation costs real training runs. Use `tm.scaled` so the cheap rungs actually get cheaper.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"   # fast unit tests
pytest                 # everything (spawns short CPU training runs, ~1 min)
```

Releases: tag `vX.Y.Z` and the `release` workflow builds the package and publishes it to PyPI via trusted publishing.

## License

MIT
