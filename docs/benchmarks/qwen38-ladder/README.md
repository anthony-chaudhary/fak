# Qwen3.8 concept hill-climb

**Verdict:** prove each candidate change cheaply on Qwen3.5, but require a paired baseline-versus-candidate win on Qwen3.8-27B for the final claim. Qwen3.8 has no official sub-27B checkpoint as of 2026-08-22. The useful smaller family is Qwen3.5: its 0.8B, 2B, 4B, 9B, and 27B checkpoints use the same `qwen3_5` runtime architecture as Qwen3.8-27B. Qwen3.5-27B also has the same 64 layers, hidden size 5120, intermediate size 17408, 24 attention heads, four KV heads, head dimension 256, vocabulary 248320, and 3:1 GDN/full-attention cadence.

## Value frame

- **For:** kernel and model-runtime developers iterating on Qwen3.8 support.
- **Problem:** each failed 27B run is slow and expensive, so implementation defects are discovered too late.
- **Today:** experiments jump directly to Qwen3.8-27B or use an unrecorded proxy whose evidence cannot support promotion.
- **Better because:** the ladder stops at the cheapest failing scale while preserving one corpus, metric, and baseline/candidate implementation pair.
- **Witness:** `fak model qwen38-ladder --selfcheck` renders the pinned ladder; an evidence packet returns `PROMOTE`, `HOLD`, or final `PASS`.

Centrality is **Enabling**. P1 managed context: one immutable corpus follows the candidate. P2 net-true efficiency: most loader, request, tokenizer, kernel-shape, and scoring failures stop at 0.8B–9B. P3 bounded adaptation: every stage says what it does and does not prove. P4 integrated operations: the sanctioned GPU nodes run the 27B rehearsal and target stages; their reports feed the existing Qwen3.8 quant campaign.

## Ladder

| Stage | Exact model | Use it to prove | Never infer from it |
|---|---|---|---|
| smoke | Qwen3.5-0.8B | loader, tokenizer/template, request shape, shape-generic kernels | quality, 27B memory, Qwen3.8 weights |
| behavior | Qwen3.5-2B | rapid tool/JSON/coding scorer iteration | target quality |
| width | Qwen3.5-4B | wider tensor shapes and behavior trend | target memory or quality |
| quality-proxy | Qwen3.5-9B | medium-scale quality signal and untied embeddings | target quality |
| scale-rehearsal | Qwen3.5-27B | exact 27B tensor geometry, depth, heads, and memory class | Qwen3.8 weight behavior or identity |
| target | Qwen3.8-27B | exact correctness, quality, performance, and release decision | anything beyond the witnessed quant/hardware envelope |

Promotion is a sequential paired experiment, not a benchmark race and not mere compatibility. At every stage, run the pinned baseline runtime and candidate runtime on the same exact model revision, corpus, prompts, seeds, quantization policy, scorer, and environment. A candidate promotes only when it:

1. meets that stage's absolute correctness floor;
2. does not regress correctness versus baseline; and
3. improves the declared metric by at least `minimum_improvement_pct` in the declared `lower`- or `higher`-is-better direction.

A failure holds at that stage so the next edit is tested at the cheapest model that reproduces it. Changing the candidate, baseline, corpus, scorer, metric, quant, or measurement environment starts a new evidence chain. Record those non-model controls in a canonical manifest and bind each result with `environment_sha256`; the evaluator rejects an unbound result.

The 27B stages belong on the sanctioned fleet nodes in `docs/fleet-compute-nodes.md`; lack of a local GPU is not a terminal blocker. Use Qwen3.5-27B as the geometry rehearsal and Qwen3.8-27B as the exact target. Do not skip the target because geometry is identical: weights, training, quality, latency distribution, and exact identity remain target-only facts.

## Commands

```bash
fak model qwen38-ladder --selfcheck
fak model qwen38-ladder --evidence evidence.json
```

Minimal first-run evidence:

```json
{
  "schema": "fak.qwen38-ladder-evidence/1",
  "concept": "fused-gdn-kernel-v1",
  "corpus_sha256": "<sha256>",
  "baseline_runtime_sha": "<git-sha>",
  "candidate_runtime_sha": "<git-sha>",
  "metric": "p95_ms",
  "direction": "lower",
  "minimum_improvement_pct": 5,
  "results": []
}
```

That returns `PROMOTE` with the pinned smoke model. Append results in ladder order. Each result binds `stage_id`, exact `model`, immutable `revision`, `environment_sha256`, `trials`, paired correctness counts, and paired metric values. `PASS` is impossible until the exact Qwen3.8-27B target result beats baseline and passes correctness.

## Upstream evidence

Observed through the Hugging Face model API and each checkpoint's `config.json` on 2026-08-22:

- Qwen3.5 releases: <https://huggingface.co/api/models?search=Qwen3.5&author=Qwen&limit=100&full=true>
- Qwen3.8 releases: <https://huggingface.co/api/models?search=Qwen3.8&author=Qwen&limit=100&full=true>
- Exact target config: <https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json>

All model revisions are pinned in `internal/qwen38ladder`. Re-query upstream before deliberately updating a stage; do not silently float revisions.

## Evidence modes and promotion policy

`fak.qwen38-ladder-evidence/1` remains backward compatible. An artifact with the
original top-level `corpus_sha256`, aggregate trial counts, and metrics is the
**concept-proof mode** used by the three-trial arithmetic spine. It can prove the
ladder mechanism and authorize only the next experiment; it is not default or
release evidence.

A repeated default/release decision uses the strengthened mode: declare a fixed
`corpora` manifest (name, task family, immutable SHA, and correctness floor), a
`confidence` rule with `method: "paired-win-rate"`, minimum paired samples, and
minimum win rate, then provide one ordered result for every corpus at every rung.
Each result binds both arms to corpus, runtime, model revision, and environment
identities and carries raw paired measurements. The evaluator fails closed with
typed reason codes for missing arms, identity drift, too few pairs, an invalid
baseline, quality regression, confidence failure, or a proxy-only default/release
claim. Only complete multi-corpus evidence at `Qwen/Qwen3.8-27B` returns `PASS`.

Promotion evidence is every declared family clearing its correctness floor,
minimum sample count, metric threshold, and paired-win-rate threshold. Demotion or
retirement evidence is any typed `HOLD`; stop before the next expensive rung and
repair or retire the candidate. The policy is invalidated if paired observations
are not comparable independent trials (for example, correlated warm-cache repeats)
or if the fixed corpus manifest no longer represents the intended workload; in
that case revise and re-hash the manifest before making another promotion claim.
