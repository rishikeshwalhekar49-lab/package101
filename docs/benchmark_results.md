# Benchmark results (bundled suite, rule-based agent)

Produced with trainmedic 0.1.0 on a laptop CPU (Apple Silicon, 4 parallel workers):

```bash
trainmedic bench build bench_suite
trainmedic bench verify bench_suite --seeds 4
trainmedic bench run bench_suite --agent rules --seeds 4 --workers 4 --out results
```

Verification labelled the 39 mutants as **25 silent failures** (the metric was worse on all 4 seeds), **14 benign**
(equivalent mutants, kept as extra negative controls) and **0 crashes**. With the 4 clean pipelines, that is 43 instances.
A buggy instance counts as *repaired* only if the agent's final patch restores behaviour that is not significantly
worse than the clean twin (checked on 4 fresh seeds), or the patched code is identical to the clean code.

# trainmedic benchmark: agent `rules`

25 buggy pipelines (silent failures), 4 clean pipelines, 14 benign mutants, 0 agent errors, 1143s total.

| metric | value |
|---|---|
| detection_rate | 0.92 |
| expected_signal_rate | 0.92 |
| repair_rate | 0.76 |
| exact_match_rate | 0.47 |
| false_alarm_rate | 0.00 |
| abstention_accuracy | 1.00 |
| damage_rate | 0.00 |
| benign_patched_rate | 0.07 |
| mean_patch_size | 1.68 |
| mean_minimal_size | 1.68 |
| hyperparameter_fixes | 3 |

## By category

| category | n | detection | repair |
|---|---|---|---|
| attention | 4 | 1.00 | 0.75 |
| data | 2 | 1.00 | 0.50 |
| loss | 1 | 1.00 | 1.00 |
| model_mode | 4 | 1.00 | 0.75 |
| model_structure | 1 | 1.00 | 0.00 |
| optimization | 12 | 0.92 | 0.92 |
| rl_post_training | 1 | 0.00 | 0.00 |

## By contamination split

| split | buggy | repair | clean+benign | damage |
|---|---|---|---|---|
| fresh | 25 | 0.76 | 18 | 0.00 |

## Per instance

| instance | label | status | repaired | exact | fix kind |
|---|---|---|---|---|---|
| mlp_classify__clean | clean | no_bug_found |  |  |  |
| mlp_classify__remove_zero_grad | silent_failure | repaired | True | False | code |
| mlp_classify__zero_grad_after_backward | silent_failure | repaired | True | False | code |
| mlp_classify__remove_eval | silent_failure | repaired | True | False | code |
| mlp_classify__remove_train | benign | candidates_rejected |  |  |  |
| mlp_classify__softmax_before_ce | benign | candidates_rejected |  |  |  |
| mlp_classify__disable_shuffle | silent_failure | repaired | True | True | code |
| mlp_classify__lr_x100 | silent_failure | repaired | True | False | hyperparameter |
| mlp_classify__remove_scheduler_step | benign | repaired |  |  | code |
| cnn_patterns__clean | clean | no_bug_found |  |  |  |
| cnn_patterns__remove_zero_grad | silent_failure | repaired | True | False | code |
| cnn_patterns__zero_grad_after_backward | silent_failure | repaired | True | False | code |
| cnn_patterns__remove_eval | benign | candidates_rejected |  |  |  |
| cnn_patterns__remove_train | silent_failure | candidates_rejected | False |  |  |
| cnn_patterns__softmax_before_ce | benign | candidates_rejected |  |  |  |
| cnn_patterns__disable_shuffle | benign | candidates_rejected |  |  |  |
| cnn_patterns__lr_x100 | silent_failure | repaired | True | True | hyperparameter |
| cnn_patterns__clip_after_step | benign | candidates_rejected |  |  |  |
| cnn_patterns__batch_mixing_flatten | silent_failure | diagnosed_only | False |  |  |
| cnn_patterns__skip_eval_normalization | silent_failure | diagnosed_only | False |  |  |
| tiny_lm__clean | clean | no_bug_found |  |  |  |
| tiny_lm__remove_zero_grad | silent_failure | repaired | True | False | code |
| tiny_lm__zero_grad_after_backward | silent_failure | repaired | True | False | code |
| tiny_lm__remove_eval | silent_failure | repaired | True | True | code |
| tiny_lm__remove_train | benign | no_bug_found |  |  |  |
| tiny_lm__softmax_before_ce | silent_failure | repaired | True | True | code |
| tiny_lm__unscale_grad_accum | benign | diagnosed_only |  |  |  |
| tiny_lm__drop_dropout_training_flag | silent_failure | repaired | True | True | code |
| tiny_lm__invert_causal_mask | silent_failure | repaired | True | True | code |
| tiny_lm__remove_causal_mask | silent_failure | diagnosed_only | False |  |  |
| tiny_lm__remove_attention_scale | benign | candidates_rejected |  |  |  |
| tiny_lm__softmax_wrong_dim | silent_failure | repaired | True | True | code |
| tiny_lm__unshift_lm_targets | silent_failure | repaired | True | True | code |
| tiny_lm__lr_x100 | silent_failure | no_bug_found | False |  |  |
| tiny_lm__clip_after_step | benign | candidates_rejected |  |  |  |
| tiny_lm__remove_scheduler_step | silent_failure | repaired | True | True | code |
| dpo_toy__clean | clean | no_bug_found |  |  |  |
| dpo_toy__remove_zero_grad | benign | candidates_rejected |  |  |  |
| dpo_toy__zero_grad_after_backward | silent_failure | repaired | True | False | code |
| dpo_toy__remove_eval | benign | no_bug_found |  |  |  |
| dpo_toy__flip_kl_sign | benign | candidates_rejected |  |  |  |
| dpo_toy__swap_preference_pair | silent_failure | no_bug_found | False |  |  |
| dpo_toy__lr_x100 | silent_failure | repaired | True | False | hyperparameter |

## Reading the misses

- **No template fix exists** for `batch_mixing_flatten`, `skip_eval_normalization` and `remove_causal_mask`.
  They are detected (by the batch-independence probe, the input-distribution invariant and the causality probe)
  and diagnosed, but generating the patch needs the LLM modes (`--mode hybrid --llm anthropic`).
- **Not detected:** `swap_preference_pair` (a purely semantic DPO bug) and `tiny_lm__lr_x100`. In the second
  case AdamW keeps every single update under the parameter norm, so the update-ratio signal stays below threshold.
  A reference run (`trainmedic infer`) or an LLM is needed.
- **Rejected by the statistical gate:** `cnn_patterns__remove_train`. The fix was proposed, but a 6.5-point
  effect was not significant on 5 held-out seeds at a reduced budget. The gate is deliberately conservative.

The LLM agents (`--agent hybrid|llm|naive-llm --llm anthropic`) have not been run here: they need API credentials
and cost money.
