# V4.1 Flash DSpark residency spec-decode port study (DS4.1 P8, ticket 3344)

Date: 2026-10-13
Receipt: `docs/benchmarks/receipts/halo-v41-100-30/v41-q5-dspark-20261013-strix3.md`
Witness: `go test ./internal/model/ -run V41DSparkResidency -count=1` (plus the owning
spec-decode suite `go test ./internal/model/ -run V41SpecDecode -count=1`)
Label: [SW-VERIFIED] (source-pinned port study; no serve, no GPU, no model load in this run)
Status: experiment / port study — no production envelope claim.

## What this is

The P8/Q5 port study for "keep the DSpark drafter resident, stream the target, and prove code
vs prose separately." It is a typed, fail-closed record in `internal/model/v41_dspark_residency.go`:
the pinned upstream anchors, the draft-resident / target-streamed residency byte plan, the
UPSTREAM-REPORTED code-vs-prose acceptance rows, and an admission seam that refuses a
single-Halo resident-DSpark claim without a passing load.

It implements no decoder, no draft model, no verify loop, and no KV plan. The deliverable's
value is the de-risked shape and the refusal, not a throughput number.

## Definition of done

- [x] SGLang DSpark + evolve line anchors pinned with licenses. The pins are carried as typed
  `V41DSparkSourcePin` values and validated by `ValidateV41DSparkPins` (full 40-hex revision,
  closed license set {Apache-2.0, MIT}, closed port class, line window).
- [x] Draft-resident / target-stream path with acceptance labelling. `V41DSparkResidency` carries
  `DraftResident` / `TargetStreamed` and the census bytes; every acceptance row carries
  `EvidenceClass == "UPSTREAM-REPORTED"`, and that class is validated so an unlabelled or
  `fak-measured` row cannot pass as this study's evidence.
- [x] Code vs prose tok/s recorded separately. Two acceptance rows keyed by `V41DSparkWorkload`
  (`code`, `prose`), with `AcceptanceFor` lookup; the spec requires both to be present.
- [x] Residency bytes recorded; no 1xHalo overclaim. `AdmitResidentClaim(nodeCount, loadPassed)`
  refuses `nodeCount == 1` with `ErrV41DSparkResidencyOverclaim` when the plan does not fit a
  one-Strix-Halo budget and no passing load is supplied, and refuses an unconfigured study at
  any node count.

## Pinned anchors (verified 2026-10-13, `gh api`)

| repo | revision | path | license | class |
| --- | --- | --- | --- | --- |
| sgl-project/sglang | `45418d7dc648d0d71501b8b905a2499a40e9b971` | `python/sglang/srt/models/deepseek_v4_dspark.py` (blob `62b5776…`) | Apache-2.0 | ADAPT |
| evolvemarketingitalia/ds4 | `5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75` | `docs/STRIX_HALO_DSPARK.md` | MIT | ADAPT |
| evolvemarketingitalia/ds4 | `5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75` | `docs/SPECULATIVE_DECODING.md` | MIT | ADAPT |
| evolvemarketingitalia/ds4 | `5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75` | `tools/strix-halo/dspark41_config.json` | MIT | ADAPT |

SGLang shape anchors (`deepseek_v4_dspark.py`, 1203 lines, at the pin):
`class DSparkV4Stage(DeepseekV4DecoderLayer)` at 558; `class DeepseekV4ForCausalLMDSpark(nn.Module)`
at 726; `class DSparkV4MarkovHead(BaseDSparkMarkovHead)` at 394; `def build_dspark_v4_confidence_head`
at 515; `self.gamma` 770 / `self.block_size = self.gamma` 774; `require_markov()` 765;
`self.num_stages` 776/778; `self.num_target_features` 786/788; `def write_target_hidden_kv` 885.

evolve ds4 doc anchors (`docs/STRIX_HALO_DSPARK.md`, 214 lines, at the pin):
tok/s table 15-19; acceptance 21-22; residency 88, 123-124; draft shape 147.

## Residency plan (census, planning figures — not fak-measured)

| Field | Value |
| --- | --- |
| Target (`TargetBytes`) | 152 GiB |
| Draft (`DraftBytes`) | ~7.4 GiB (3 stages, block size 5, Markov rank 256, target layers 37-39, 128 experts) |
| Resident total | ~159.4 GiB |
| One-Strix-Halo host budget | 128 GiB (per node) |
| Minimum nodes | 2 |

The census shape does not fit one 128 GiB Strix Halo, so the study's own admission seam refuses
a 1-node resident claim unless a real passing load is supplied. That is the "no 1xHalo
resident-DSpark claim without a passing load" constraint, expressed as a typed refusal rather
than prose.

## Acceptance rows (UPSTREAM-REPORTED — NOT fak-measured)

| Workload | Baseline tok/s | DSpark tok/s | Accepted/cycle | Acceptance rate | Byte-identical |
| --- | ---: | ---: | ---: | ---: | --- |
| code (143-line C edit) | 13.8 | 31.7 | 3.6 | 0.975 | upstream-reported |
| prose (Italian analysis, 256 tok) | 13.4 | 16.6 | 1.5 | 0.55 | upstream-reported |

The code gain is larger than the prose gain, which is the ticket's core finding. Every row is
labelled `UPSTREAM-REPORTED`, and the validator refuses any row that is not, so no reader and no
consumer can mistake these for a measured fak number.

## Witness

```
$ go test ./internal/model/ -run V41DSparkResidency -count=1
ok  	github.com/anthony-chaudhary/fak/internal/model

$ go test ./internal/model/ -run 'V41DSpark|V41SpecDecode' -count=1
ok  	github.com/anthony-chaudhary/fak/internal/model
```

The test suite (authored in a separate context from the implementation) covers: the default study
validating clean; the four pins; single-field pin tampering (revision case/length, license set,
port class, line window, blob presence); `ResidentBytes`/`FitsHost` arithmetic; the default
residency bytes and flags; the acceptance rows including exact ticket constants and the
code-gain > prose-gain finding; every `Validate` error arm (missing row, zero/NaN/Inf rates,
non-improving DSpark rate, out-of-range acceptance rate, missing/wrong evidence class); and the
full `AdmitResidentClaim` ladder (nodeCount 0/negative, 1-node non-fitting refused with
`errors.Is`, 1-node fitting admitted, 1-node load-passed admitted, 2-node admitted, zero-value
study refused at 1 and at 2).

## Reconciliation (from the earlier admitted base)

On the same box, strix3 slice-1 (`670e1db8cc3`, ring 12G) reported warm decode 0.771-0.794 tok/s;
peer update 2026-10-12 reports a GTT auto 64 GiB ring at warm 1.28-1.30 tok/s (P1 superseded by
public `164ec3575b2`), and P2 step 1 landed public `cc7566f1016` (cold 0.648 -> 0.780 tok/s).
This study's acceptance rows are the upstream yardstick (13.8-14.0 tok/s exact decode), not a
fak measurement, and no fak DSpark number is claimed here.

## Still open / next checkable steps

1. This is a shape and refusal artifact, not a served path. The next step is the P3 batched
   routed dispatch (private P3) that the draft-resident / target-streamed verify loop depends on.
2. A two-node resident DSpark bring-up can then exercise `AdmitResidentClaim(2, true)` and replace
   the UPSTREAM-REPORTED rows with `[HW-WITNESSED]` rows at the same schema.
3. Policy sim (Q3/P6) and CPU-miss (Q4/P7) stay out of scope per the ticket's gold-plating boundary.

## Provenance

- Implementation: `internal/model/v41_dspark_residency.go` (new).
- Tests: `internal/model/v41_dspark_residency_test.go` (new; authored in a distinct context from
  the implementation).
- Third-party notices: `THIRD_PARTY_NOTICES.md` (SGLang Apache-2.0 and evolve ds4 MIT entries).
- Ticket: private#3344; parent engine frontier public#13668; P-index
  `docs/tickets/ds41-oss-expert-streaming-ports-20261012/` (P8) and
  `docs/research/STUDY-ds41-perf-expert-streaming-offload-2026-10-13.md` (Q5).
