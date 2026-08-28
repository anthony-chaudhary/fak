# Qwen trajectory snapshot dogfood — 2026-08-27

Status: **VERIFIED after #9702 and #9703; the historical pre-fix capture remains REFUTED below.**

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

## Post-fix retained rerun — 2026-08-28

The combined #9681, #9702, and #9703 base completed one capture and one replay over the same private corpus. Both passes reported `verified=true`, 152 files, 150 canonical sessions, and zero refused records. The corpus digest was `89e1a483b036bd734af91d7bf23d3a15b784a495cd04725626b120d3f9091542`. Capture and replay were byte-identical:

- JSONL SHA-256: `33bd683c18c7268b464e0a5909261d0a2b6aa8f978c8b7a23874f66e76cd02ca`
- Markdown SHA-256: `701280bdb98277523a9f1533cdf8a0a75796bc38c9bf78ce0b2f622cd81da961`

The private snapshot used restricted permissions: `0700` for the snapshot directory and `0600` for its manifest. No raw corpus bytes are retained here.

The public-safe readback immediately preceding that capture selected the same 150 canonical sessions from 152 fragments. It accounted for 108,393,944 input, 2,618,842,368 cache-read, zero cache-create, and 8,310,616 output tokens: 2,735,546,928 total. The top ten held 1,820,248,183 tokens, or **66.5405577%**. Continued writes to those live sessions advanced the later pinned capture to 24,011 exact usage records and 66.63808296%; this is expected sliding-corpus movement, not capture/replay drift. The two outputs from the pinned corpus remained byte-identical.

### Corrected top-ten attribution

`Turns` counts Codex task-start events. Every current top-ten session had an operator-authored Qwen objective. All had zero mutation churn; only rank 1 had one repeated failed action. Tool-error counts alone do not prove avoidable work.

| Rank | Session | Input / cache-read / output | Turns / tools / errors / churn / repeated | Classification | Public landed effect | Conservative counterfactual |
|---:|---|---:|---:|---|---|---:|
| 1 | `01a03b0a-0981-7ba3-ae80-7fd5402239cd` | 5,400,182 / 373,675,776 / 458,884 | 2 / 3,002 / 44 / 0 / 1 | Useful Qwen Metal residency/runtime work | `84760f5a25a8`, `f732527d4daf` | 0 auditable tokens; the repeat has no token attribution |
| 2 | `01a03b37-7f33-7050-9fd7-0cf89564dd56` | 5,953,198 / 316,454,144 / 815,441 | 35 / 2,354 / 72 / 0 / 0 | Useful Qwen native profiling | `425a71bb5616` | 0 |
| 3 | `01a03c91-d142-7133-9fab-3260bc53bf21` | 5,701,131 / 271,320,576 / 772,657 | 4 / 2,126 / 128 / 0 / 0 | Useful Qwen Mac closure/evidence work | `4212905035a4`, `3e5891679d4b` | 0 |
| 4 | `01a03b13-e4d3-7021-8364-b09832f30fe1` | 5,346,077 / 222,344,448 / 690,023 | 41 / 1,713 / 52 / 0 / 0 | Useful Qwen GDN work | `dfa6ea83b927`, `254fb97a797f` | 0 |
| 5 | `01a03b37-4676-7c62-9e60-d5083d511db9` | 4,425,036 / 166,486,528 / 491,948 | 22 / 1,301 / 28 / 0 / 0 | Useful Qwen runtime/cold-arm work | `6abd002e0dec` | 0 |
| 6 | `01a04401-0fe0-7c23-aa3c-ec8bbf47cbbc` | 3,208,196 / 166,559,488 / 328,865 | 2 / 1,301 / 71 / 0 / 0 | Useful Qwen batch/GDN wave | `d7ce989e497d`, `2f0fcd0f5353` | 0 |
| 7 | `01a04678-03a4-7840-9f61-1d9a861d7662` | 1,623,761 / 97,444,608 / 180,423 | 1 / 797 / 38 / 0 / 0 | Useful Qwen issue/closure wave | `1f7aebc59e7c`, `8f070581cb4b` | 0 |
| 8 | `01a03c93-fe4a-7cf1-85ee-1b6f26129431` | 2,628,481 / 67,498,496 / 320,549 | 22 / 513 / 11 / 0 / 0 | Useful Mac/Qwen performance wave; some work remained unlanded | `3d90878da`, `4e3d17b92`, `bc88a3817` | 0 |
| 9 | `01a03eec-abcf-7eb0-9ae7-1c63f48ffb5b` | 1,327,250 / 50,507,776 / 130,146 | 1 / 404 / 7 / 0 / 0 | Useful but incomplete exact Qwen3.8 decode-parity campaign | None independently verified | 0 |
| 10 | `01a0451f-5952-7643-8108-90808bd16212` | 9,693,054 / 38,136,576 / 324,465 | 8 / 1,325 / 83 / 0 / 0 | Useful Qwen/GLM hardware and long-context economics | `02ed431a6604`, `49433e5b5f81`, `6667e4e2e66e` | 0 |

The operational saving lower bound remains zero. The current top ten is productive work, not demonstrated replay, churn, or error-loop waste. The concentration percentage is therefore diagnostic, not an optimization target.

### Concentration reconciliation

- **Historical 35.47%:** 450 sessions accounted for 2,864,577,361 tokens. The rounded fraction implies approximately 1,016,065,590 top-ten tokens, but the exact numerator was not retained and two malformed duplicate-usage records blocked broad conclusions.
- **Pre-fix 43.41%:** the 2026-08-27 live fallback counted 430 canonical sessions, 5,160,285,455 total tokens, and 2,240,250,063 top-ten tokens. It preceded #9703 and included injected-guidance matches. Four proven false positives alone contributed 689,817,465 tokens, or 13.37% of that denominator.
- **Post-fix 66.5405577%:** #9703 reduced the selected cohort to 150 canonical sessions and removed all four known top-ten contaminants. Relative to the pre-fix run, the denominator fell 46.99% while the top-ten numerator fell 18.75%. The resulting increase is a purified topical denominator concentrated around legitimate long Qwen sessions, not increased waste.

### Mitigations and remaining gaps

- #8714 is closed by `f6d64aa37`: select user-authored topical content. #9703, closed by `8f070581cb`, completes the later-discovered injected-guidance exclusion.
- #8715 is closed by `bd638fc69`: report deterministic top-contributor concentration.
- #8716 is closed by `cab2b8c00`, `257e3f3bcb`, and `3e27e1e4b9`: stop a third identical tool failure while allowing changed attempts and recovery.
- #8700 is closed by `d4cb83dcdb`: rank tool-error families with repeated-failure and churn attribution.
- #8702 is closed by `86db073803`: parse Codex hook durations with typed missing and malformed states.
- #8703 is closed by `1b982cc711`: enforce canonical Qwen input-amplification budgets.
- #8717 is closed by `b7930a79cc`: detect and bound mutation churn.
- #8718 is closed by `26fe15c497`: fail closed on unsupported evidence shapes.

No original #8929 gap remains open. The highest-confidence improvement is the combined measurement spine: #9681 (`1f7aebc59`) pins and verifies the private corpus, #9702 (`1ae91f630`) makes its output deterministic, and #9703 (`8f070581cb`) excludes injected guidance from topical admission. It corrects what the audit measures without inventing a performance saving or suppressing productive long work.
