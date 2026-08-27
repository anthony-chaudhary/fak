# vLLM forge-corpus classification at the August 26, 2026 cutoff

This directory is the commit-sized receipt, summary, and mechanism-cluster index
for GitHub issue #9274. It was generated from the immutable
`vllm-project/vllm` forge corpus at revision
`f18d0ba90d972a852a351c98be3f42b31372cfe4` and cutoff
`2026-08-26T22:35:00Z`.

## Receipt

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Input corpus, not committed here | 252,354,769 | `2a66d4876aee3811eb200c0884c6558a5f3ac86c90b6c7f8b92f45b85fe671b2` |
| Validated full classification, retained in worker scratch | 155,816,295 | `602fbea71cb28a54cd509209245961996f5a55dc76754c51732d497ddf1944bd` |
| [`index.json`](index.json) | 1,515,970 | `b62813994a053eba39e85a817c0e2f68f0401ae9f6023cde66bb9684d6b7ddaa` |
| [`classification.schema.json`](../../../internal/studyclass/classification.schema.json) | 7,233 | `a1ee4a176ef353ea2199e29ca78fb878cdf6d79f0d1aa5b772e0363a1f833963` |

The corpus receipt's normalized-record checksum is
`sha256:75577e65370ac201f29efcf6ce38c15b7bf48c0d793fbfb946407a4045de7c14`.
`index.json` binds all of these facts, the rules schema, the full records and
clusters checksums, and a checksum for each compacted cluster membership list.

The cutoff is inclusive on `created_at`; the source crawl is not a historical
state snapshot. Exactly 49 retained records have `updated_at` after the cutoff,
which is recorded as `input.post_cutoff_updated_records`. The upstream receipt
also marks its mixed-issues versus dedicated-pulls identity relation as
unavailable for this legacy count-only crawl. The classifier therefore records
only explicit field or text evidence and never invents duplicate, supersession,
or implementation relationships.

## Summary

- **Records:** 53,848, each with exactly one primary disposition.
- **Mechanism clusters:** 193.
- **Sources:** 35,551 pulls; 17,606 issues; 489 discussions; 104 releases;
  63 labels; 35 milestones.
- **Disposition:** 20,400 merged/landed; 10,668 closed-unmerged; 9,186
  stale/superseded; 7,099 regression/bug; 4,227 open proposal; 1,992
  support/question; 202 release/metadata/non-candidate; 74 duplicate.
- **State:** 46,193 closed; 7,488 open; 167 without a state because they are
  release or label metadata.
- **Disposition confidence:** 38,429 high; 14,371 low; 1,048 medium.

Mechanism tags are non-exclusive, so their counts do not sum to the record
count:

| Mechanism | Records |
|---|---:|
| model/backend/hardware | 16,918 |
| tests/CI/docs | 12,715 |
| architecture/runtime | 10,233 |
| KV/cache | 7,574 |
| APIs/tool calling/structured output | 7,405 |
| kernels/compilation | 5,142 |
| observability/operations | 3,800 |
| speculative decoding | 3,299 |
| scheduling/batching | 2,088 |
| distributed/parallelism | 2,015 |
| memory/residency | 1,310 |
| reliability/security | 1,245 |
| explicit non-candidate | 202 |

## Reproduce

```bash
fak study-classify classify \
  --corpus /path/to/vllm-corpus-2026-08-26T2235Z.json \
  --out /allocated/scratch/vllm-classification.full.json \
  --index-out docs/research/vllm-classification-2026-08-26/index.json \
  --related-limit 8

fak study-classify validate \
  --classification /allocated/scratch/vllm-classification.full.json \
  --corpus /path/to/vllm-corpus-2026-08-26T2235Z.json

fak study-classify validate-index \
  --index docs/research/vllm-classification-2026-08-26/index.json \
  --classification /allocated/scratch/vllm-classification.full.json \
  --corpus /path/to/vllm-corpus-2026-08-26T2235Z.json
```

`related` means shared deterministic mechanism evidence only. It does not mean
that GitHub records link to, duplicate, supersede, close, or implement one
another.
