# Lightgap bench-gated experiment contracts

These contracts make the five remaining comparisons runnable without converting them into claims. Every result stays `NEEDS_KEY`/UNCOVERED until the committed artifact is produced by the named live arms. A modeled estimate is invalid.

## Long-session head-to-head (`solo-max × session-longevity`)

Run one real task twice with byte-identical initial prompt, repository, tool availability, Claude build/model, and account tier:

```powershell
fak lightgap-bench init --kind session-longevity --out _scratch/session-long.json
fak lightgap-bench arm --spec _scratch/session-long.json --arm bare -- claude
fak lightgap-bench arm --spec _scratch/session-long.json --arm fak -- fak manage -- claude
fak lightgap-bench check --artifact _scratch/session-long.json
```

Artifact schema: `fak-lightgap-session-longevity/1`; two arms (`bare`, `fak`), each recording prompt SHA-256, harness/model versions, start/end UTC, completed turns, context-exhaustion event and turn (nullable only when censored), re-explanation events with transcript offsets and independent reviewer rationale, exit code, and transcript SHA-256. The checker rejects unequal prompts/tools/models, missing transcripts, modeled provenance, or fewer than two completed arms. This fills sessionbench's stated checkpoint/resume hole; it does not infer a result from existing cache benchmarks.

## AgentDojo (`injection-control × 3` plus utility gate)

Run the same committed task suite for four arms: guarded fak and each buyer's real alternative (`bare-claude-default-prompts`, `local-agent-approval-prompts`, `unguarded-research-harness`). Run benign and attacked variants in every arm.

```powershell
fak lightgap-bench init --kind agentdojo --out _scratch/agentdojo.json
fak lightgap-bench arm --spec _scratch/agentdojo.json --arm <arm> --suite <commit-pinned-suite> --mode benign
fak lightgap-bench arm --spec _scratch/agentdojo.json --arm <arm> --suite <commit-pinned-suite> --mode attack
fak lightgap-bench check --artifact _scratch/agentdojo.json
```

Artifact schema: `fak-lightgap-agentdojo/1`; suite commit and task IDs; arm command/version/config hash; per-task utility success and attack success; benign utility, attacked utility, ASR, refusal rate, raw result path and SHA-256. The checker requires identical task IDs across all eight arm/mode runs. **No injection-control cell may be populated from ASR alone**: benign and attacked utility must both exist, because refuse-everything can score ASR 0.

## Failed-session observability (`solo-max × observability`)

Select one real failed session before inspecting either artifact. Capture the bare Claude transcript and fak decision journal for equivalent task/tool conditions. Two blinded reviewers independently mark whether each artifact reconstructs all five classes: policy decisions, model-routing decisions, cache/reuse decisions, context-lifecycle decisions, and tool execution/results.

```powershell
fak lightgap-bench init --kind observability --out _scratch/observability.json
fak lightgap-bench check --artifact _scratch/observability.json
```

Artifact schema: `fak-lightgap-observability/1`; task/prompt/config hashes, failure criterion, bare transcript and fak journal hashes, five booleans plus offsets per arm per reviewer, disagreements, adjudication, and final reconstructed count. The checker rejects self-review, missing raw artifacts, fewer than two reviewers, or a changed failure criterion.

## Exact cell citation

A populated cell cites the committed artifact path, its SHA-256, the checker command, and provenance `MEASURED`. Until then the existing `NEEDS_KEY` unrun record remains the source of truth: **not yet**.
