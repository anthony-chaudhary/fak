# Micro-context S8i: independently adjudicated semantic residual

**Status:** live two-model adjudication plus deterministic consensus/abstention fold<br>
**Issues:** #6124, parent #6033; live comparative matrix #6110<br>

## Why this corpus exists

S8f is leakage-controlled but every original answer is a structural projection. S8i adds a small semantic layer without hiding authentic source metadata or pretending deterministic facts require an LLM.

The layer asks three questions per issue:

1. Does the operational category follow literally, require semantic interpretation, or remain ambiguous?
2. Is no extra evidence needed, would read-only repository evidence help, or is current state required?
3. Is the issue actionable, not actionable, or should the grader abstain?

These are not literal copies of state, labels, or timestamps. They can expose where model interpretation and evidence-routing differ.

## Leakage and selection boundary

`experiments/microcontext/s8i-semantic-packet-2026-08-10.json` selects 16 tune and 16 held-out test records from the frozen S8f public corpus. Selection is deterministic: records need a body of at least 300 bytes, then the lowest `title_sha256` values are chosen independently inside tune and test. Thus selection cannot inspect semantic answers.

The packet retains title, body, URL, number, and split. Keeping source metadata visible is intentional: exact filters remain a fair component of every candidate pipeline. No semantic label, rationale, adjudicator identity, or answer digest appears in candidate input.

Grouped S8f split controls remain inherited. S8i introduces no record movement and its fold checks for cross-split duplicate IDs and missing judgments.

## Independent live adjudication

Two different live model routes adjudicated the same packet independently:

- sanctioned compute endpoint: `qwen2.5:14b`;
- separate OpenAI-compatible endpoint: `gpt-5.6-sol`.

Both used the same versioned task definition and zero-temperature/low-temperature structured output, but had different providers/models and did not see each other's output. Artifacts bind the packet SHA and record endpoint-neutral provenance plus usage:

- `s8i-adjudicator-a-2026-08-10.json`: 18,796 tokens;
- `s8i-adjudicator-openai-2026-08-10.json`: 23,053 tokens.

The earlier `s8i-adjudicator-b-2026-08-10.json` is retained only as a diagnostic self-agreement witness: the local endpoint ignored the attempted variant and replayed identical outputs. It is **not** accepted by the consensus verifier and is not an independent answer source.

## Agreement and abstention policy

The deterministic fold accepts a field only when both independent adjudicators agree exactly. A disagreement becomes typed `abstain`; it is never resolved by secretly choosing the preferred model.

| Field | Agreement |
|---|---:|
| Actionability | 93.75% (30/32) |
| Semantic need | 40.625% (13/32) |
| Tool need | 21.875% (7/32) |

Low agreement is a result, not noise to erase. It shows that “should a tool run?” and even “is interpretation required?” need sharper contracts or additional adjudication. The gold artifact contains 32 answers, five agreed semantic records, and 32 records with at least one abstained field.

The held-out test split contains:

- 16 records;
- three agreed semantic residuals;
- 16 records with at least one abstention;
- two agreed read-only-tool cases and one agreed current-state-tool case across the full packet.

## Blind grader

The candidate submission contains only IDs and predicted semantic/tool/actionability fields. The hidden gold bundle is supplied separately to `-semantic-grade-gold`. The grader scores one named split and requires exact agreement including abstentions.

The committed oracle-format submission proves the grader door and schema—not candidate model quality. Its held-out grade is 16/16 exact, with three semantic residuals and 16 abstention-bearing records.

## Reproduce

```powershell
# Deterministically select the blinded packet.
go run ./cmd/microcontextdemo `
  -semantic-packet-corpus experiments/microcontext/s8f-github-issues-public-2026-08-09.json `
  -semantic-packet-output experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -semantic-per-split 16

# Run each adjudicator independently. API keys are environment-expanded by the caller.
go run ./cmd/microcontextdemo `
  -semantic-adjudicate-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -semantic-adjudicate-output OUT.json `
  -semantic-endpoint ENDPOINT `
  -semantic-api-key API_KEY `
  -semantic-model MODEL `
  -semantic-adjudicator INDEPENDENT_ID

# Fold exact agreement and preserve disagreement as abstention.
go run ./cmd/microcontextdemo `
  -semantic-fold-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -semantic-fold-a experiments/microcontext/s8i-adjudicator-a-2026-08-10.json `
  -semantic-fold-b experiments/microcontext/s8i-adjudicator-openai-2026-08-10.json `
  -semantic-gold-output experiments/microcontext/s8i-semantic-gold-2026-08-10.json

# Grade a held-out submission without exposing gold to the candidate.
go run ./cmd/microcontextdemo `
  -semantic-grade-gold experiments/microcontext/s8i-semantic-gold-2026-08-10.json `
  -semantic-grade-submission SUBMISSION.json `
  -semantic-grade-output GRADE.json `
  -semantic-grade-split test
```

`-verify-semantic-gold` rejects same adjudicator/model identity, missing judgments, split contamination, zero semantic residual, or zero abstention. `-verify-semantic-grade` requires a non-empty exact held-out pass containing both residual and abstention cases.

## What this proves—and does not

S8i fixes the zero-residual benchmark defect: the held-out contract now has facts that deterministic metadata projection cannot simply emit, and uncertainty is represented explicitly. It also exercises positive-value read-only/current-state routing labels.

It does **not** yet tune or run retrieval, long-context, chunk map-reduce, and micro-context candidates on those labels. Thirty-two records are enough to establish the missing mechanism and expose major disagreement, not enough for a broad production-quality claim. #6110 must now run the eligible alternatives and capture endpoint tokens, dollars, TTFT/tail, retries, batching/cache provenance, and tool costs. A follow-on should sharpen the tool-need rubric or add a third independent adjudicator because 21.875% agreement is too low for an unqualified production gold claim.
