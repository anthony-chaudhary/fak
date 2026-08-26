# Tokenizer native-vs-alternative comparison witness

Date: 2026-08-09  
Capability: `tokenization`  
Status: **INCOMPLETE**

## Arms

- `fak native`: heap-based incremental BPE merge in `internal/tokenizer`.
- tuned incumbent: exhaustive adjacent-pair rescanning on the same pretokenized symbols.
- next-best external: `llama.cpp` tokenizer for the identical GGUF vocabulary/model.
- equivalent first-class integration: Hugging Face `tokenizers` using the identical `tokenizer.json` artifact.

The repository already has independently captured token-ID parity fixtures for llama.cpp (`oracle_qwen_test.go`, `oracle_ggml_test.go`) and Hugging Face fixture provenance in `tokenizer_test.go`. Those prove selected correctness cases, not process-level performance.

## Local merge benchmark

Host: Windows amd64, AMD Ryzen 9 9950X. Command:

```text
go test ./internal/tokenizer -run '^$' -bench '^BenchmarkBPE(Incremental|Naive)$' -benchtime=50x -count=5 -benchmem
```

Medians across five runs:

| Symbols | Arm | ns/op | B/op | allocs/op |
|---:|---|---:|---:|---:|
| 64 | native incremental | 8,036 | 12,656 | 171 |
| 64 | naive exhaustive | 6,360 | 5,344 | 176 |
| 256 | native incremental | 27,840 | 49,696 | 647 |
| 256 | naive exhaustive | 25,236 | 22,160 | 654 |
| 1,024 | native incremental | 137,780 | 198,848 | 2,537 |
| 1,024 | naive exhaustive | 101,704 | 113,968 | 2,547 |
| 4,096 | native incremental | 823,364 | 765,635 | 10,125 |
| 4,096 | naive exhaustive | 1,000,688 | 399,446 | 10,138 |

The native heap algorithm crosses ahead only at the largest fixture here and uses more bytes at every size. This is an observed local algorithm result, not a general tokenizer win.

`CompareLocal` adds a frozen same-corpus correctness check and keeps the external arms explicit. It cannot report complete while live llama.cpp and Hugging Face process latency, peak RSS, and total cost remain absent.

## Honest verdict

No net-true winner is established. Completion requires one model artifact and corpus to be run through all arms, with exact token IDs/decoded bytes, tokens per second, latency distribution, peak RSS, initialization cost, and total cost. The local benchmark and fixture oracles are partial witnesses only.
