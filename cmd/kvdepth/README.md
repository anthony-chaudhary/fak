# Reusable-prefix depth campaign

`kvdepth` measures how deep a shared prompt prefix remains reusable and
where reuse falls off under capacity pressure. It keeps four claims separate:

- `semantic_prompt_equal`: the full cold/warm prompt has the same meaning;
- `token_prefix_equal`: the declared prefix token sequence is byte-for-byte equal;
- `backend_kv.admitted`: the backend says it admitted the request prefix;
- `backend_kv.cached_input_tokens`: reuse was actually observed.

Unsupported backend fields are omitted from JSONL. The analyzer will still fold
TTFT dispersion, but it reports the reusable boundary and pressure recovery as
`unknown`; absence never becomes a zero-token cache observation.

## Deterministic witness

From the repository root:

```powershell
go run ./cmd/kvdepth -manifest cmd/kvdepth/testdata/campaign.json -selfcheck
```

The selfcheck runs two synthetic fixtures across six depths, two suffix patterns,
two turn counts, two concurrency values, three repetitions per arm, and baseline /
pressure / recovery phases. The known fixture reports 8,192 tokens as the deepest
reliably reusable prefix and a cliff at 12,288; the metric-free fixture reports the
boundary as unknown.

The checked-in request-level evidence and captured report are reproducible:

```powershell
go run ./cmd/kvdepth -manifest cmd/kvdepth/testdata/campaign.json -emit-fixtures cmd/kvdepth/testdata
go run ./cmd/kvdepth -manifest cmd/kvdepth/testdata/campaign.json -observations cmd/kvdepth/testdata/known-cliff.jsonl
```

## Live campaign input

Copy the manifest and replace every fixture pin with the exact backend, runtime,
model, fak, tokenizer, and prompt-template revisions used by the run. Each JSONL
row repeats those pins and records one request's prompt tokens, TTFT, ordering,
reset procedure, useful-work outcome, equality evidence, and optional backend KV
fields. The analyzer refuses undeclared coordinates, pin drift, incomplete
cold/warm pairs, fewer than three repetitions, and an observed envelope that does
not exercise all declared axes.

This first envelope intentionally covers one backend and one model. It does not
compare eviction policies, tune kernels, or infer undocumented allocator state.
