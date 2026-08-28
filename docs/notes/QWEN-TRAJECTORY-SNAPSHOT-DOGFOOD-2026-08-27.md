# Qwen trajectory snapshot dogfood — 2026-08-27

Status: **REFUTED for capture/replay identity; CONFIRMED for the live content-free audit.**

This is the public-safe readout for #9697 and advances #8929. It exercised the snapshot spine at `1f7aebc59e7cafaaa55ffe2bf6a8b60357756181` from `origin/main` at `e2db29f84963679e574a9174dbf08ef09ae2c857`. Raw transcripts, manifests, and full reports stayed in private allocated scratch. This note contains no transcript text, tool arguments, tool outputs, or machine-local paths.

## Commands and result

The private scratch variable below intentionally replaces the machine-local absolute path. Both capture attempts used this command shape with distinct empty snapshot targets:

```bash
PRIVATE_SCRATCH=<allocated-private-scratch>
go run ./cmd/fak trajectory audit \
  --since 7d \
  --user-contains qwen \
  --snapshot-out "$PRIVATE_SCRATCH/snapshot-a" \
  --jsonl "$PRIVATE_SCRATCH/capture-a.jsonl" \
  --md "$PRIVATE_SCRATCH/capture-a.md"
```

Both independent attempts copied the selected inputs into private staging and then failed their built-in verification with the same typed result:

```text
TRAJECTORY_SNAPSHOT_REFUSED SNAPSHOT_OUTPUT_CHANGED
trajectory audit snapshot: SNAPSHOT_OUTPUT_CHANGED: current audit output does not match the captured audit schema
```

The fail-closed cleanup removed each staging snapshot. Consequently no manifest or corpus digest was published, and the required replay command could not safely run:

```bash
go run ./cmd/fak trajectory audit \
  --snapshot "$PRIVATE_SCRATCH/snapshot-a" \
  --jsonl "$PRIVATE_SCRATCH/replay-a.jsonl" \
  --md "$PRIVATE_SCRATCH/replay-a.md"
```

Capture success was 0/2, replay success was 0/0, and capture/replay byte identity is **not established**. This does not satisfy the issue's 100% capture/replay and zero-drift operating envelope.

The content-free live fallback was run separately:

```bash
go run ./cmd/fak trajectory audit \
  --since 7d \
  --user-contains qwen \
  --jsonl "$PRIVATE_SCRATCH/qwen-live-final.jsonl" \
  --md "$PRIVATE_SCRATCH/qwen-live-final.md"
```

It completed with 430 canonical sessions from 432 raw fragments, 311,053 records, 48,033 exact usage records, and zero parse refusals. Accounted usage was 259,086,472 input, 4,884,324,480 cache-read, and 16,874,503 output tokens. The top ten held 2,240,250,063 tokens, or 43.41% of the 5,160,285,455-token cohort. Private output identities were:

- JSONL SHA-256: `9528208ddf7e13fd530dce1794452f16e06b54ee6df2493e5e2a2f03797480f3`
- Markdown SHA-256: `0c8148fdc992a38fa75f5f52a0d5f7c7613dcf0c02fdb7260b323ef285827b07`

## Top-ten attribution

`Turns` counts Codex task-start events. `Errors` is the audit's tool-error count; expected bounded waits are listed separately because they are not failures. No top-ten session had mutation churn. A public commit is named only when its subject and ancestry on current `origin/main` independently witness the effect.

| Rank | Session | Input / cache-read / output | Turns | Tools / errors / expected waits | Classification | Public landed effect | Conservative counterfactual |
|---:|---|---:|---:|---:|---|---|---:|
| 1 | `01a03b0a-0981-7ba3-ae80-7fd5402239cd` | 5,400,182 / 373,675,776 / 458,884 | 2 | 3,002 / 44 / 1,100 | Productive long Qwen work; one repeated failed action, zero churn | Q8 Metal residency `84760f5a25a8`; mapped Q4_K owner `f732527d4daf` | 0 auditable runtime tokens; the duplicate action has no attributed token cost |
| 2 | `01a03b37-7f33-7050-9fd7-0cf89564dd56` | 5,953,198 / 316,454,144 / 815,441 | 35 | 2,354 / 72 / 47 | Productive long Qwen native-profile work; zero replay/churn | Metal session counters `ca8993c3a913`; control profile `425a71bb5616` | 0 |
| 3 | `01a03c91-d142-7133-9fab-3260bc53bf21` | 5,701,131 / 271,320,576 / 772,657 | 4 | 2,126 / 128 / 97 | Productive Qwen closure and Mac-evidence work; zero replay/churn | Streamed Q4_K load ablation `4212905035a4`; slab rejection witness `3e5891679d4b` | 0 |
| 4 | `01a03b13-e4d3-7021-8364-b09832f30fe1` | 5,346,077 / 222,344,448 / 690,023 | 41 | 1,713 / 52 / 4 | Productive Qwen GDN work; zero replay/churn | Resident GDN primitive `dfa6ea83b927`; hybrid GDN route `254fb97a797f` | 0 |
| 5 | `01a0400d-a03f-7650-86a0-ca090f569732` | 30,143,680 / 191,682,048 / 853,910 | 25 | 2,680 / 104 / 0 | **Cohort defect**: productive research work, not Qwen execution | Research envelopes #9362–#9387, including `255b34e33060` and `b3d717c8412a` | 0 runtime; remove 222,679,638 from the Qwen denominator |
| 6 | `01a0400c-33a1-7240-a36c-96180b009ad7` | 15,337,949 / 167,598,976 / 411,740 | 7 | 2,059 / 169 / 0 | **Cohort defect**: productive study-forge work, not Qwen execution | Disabled-discussions terminal state `1056b5c744c2`; delta policy `3752e60701b0` | 0 runtime; remove 183,348,665 from the Qwen denominator |
| 7 | `01a03b37-4676-7c62-9e60-d5083d511db9` | 4,425,036 / 166,486,528 / 491,948 | 22 | 1,301 / 28 / 33 | Productive long Qwen runtime work; zero replay/churn | Measured Qwen3.8 cold-arm runner `6abd002e0dec` | 0 |
| 8 | `01a04401-0fe0-7c23-aa3c-ec8bbf47cbbc` | 3,208,196 / 166,559,488 / 328,865 | 2 | 1,301 / 71 / 177 | Productive Qwen batch/GDN wave; zero replay/churn | Qwen native batch repair `d7ce989e497d`; duplicate encoder removal `2f0fcd0f5353` | 0 |
| 9 | `01a04403-68e1-7173-b48c-b2ba235e7af8` | 1,791,649 / 141,595,392 / 189,943 | 2 | 1,124 / 39 / 168 | **Cohort defect**: productive code-slop work, not Qwen execution | Benchmark helper refactors `b1539ebc2050` and `e73e7065a868` | 0 runtime; remove 143,576,984 from the Qwen denominator |
| 10 | `01a04401-b3ff-7fb2-831d-6f9509c59265` | 2,404,652 / 137,517,568 / 289,958 | 11 | 1,112 / 64 / 87 | **Cohort defect**: productive guard work, not Qwen execution | Guard-route witnesses `67a5dd2e2d18`, `534c481d422d`; provider fold `13d33528b9f6` | 0 runtime; remove 140,212,178 from the Qwen denominator |

The operational saving lower bound is zero: the evidence shows durable landed effects, the audit assigns zero tokens to its one repeated-failure event, and tool-error counts alone do not prove avoidable work. The exact cohort correction is different: excluding the four proven false positives removes 689,817,465 accounted tokens, 13.37% of the current selected denominator, without claiming that their useful execution should not have happened.

## Defects filed

- #9702 — snapshot output digest changes between two audits of the same staged bytes. Marker: `fanout-trajectory-spine-00cac68f178d67cb-dogfood-self-run-output-drift`.
- #9703 — injected repository guidance can satisfy `--user-contains` and contaminate a topical cohort. Marker: `fanout-trajectory-spine-00cac68f178d67cb-dogfood-self-run-topical-injected-context`.

The smallest terminal path for #9697 is #9702 first, then repeat the two capture/replay cycles and append the corpus/output digests and zero-byte drift result here. #9703 is separately required before #8929 can treat this Qwen cohort as a defensible optimization denominator.
