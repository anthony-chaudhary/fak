# Qwen3.8-27B BF16 reference — issue #8308

**Verdict: PROMOTE as the quality oracle.** The pinned BF16 checkpoint passed all 18
trials (six frozen workload families, three repetitions each) on one A100-80GB with no
fallback. The run loaded in 75.336 seconds, delivered 17.339 completion tokens/second
across the campaign, peaked at 67.797 GiB and 319.29 W, survived restart/readiness, and
returned to 536 MiB after cleanup.

This is an **Enabling** result for operators comparing Qwen3.8 deployment arms. The
problem was that a quantized arm had no immutable full-precision quality and resource
ceiling. The next-best alternative was an unpinned one-off BF16 serve, which could not
detect checkpoint, template, fallback, workload, or lifecycle drift. This witness is
better because one validator-bound archive fixes those variables and makes every later
arm comparable and rejectable.

The four project checks are covered directly: the long-context and repeated-cache
fixtures exercise managed context (P1); quality gates the measured latency, memory, and
power data (P2); immutable identity and a stale date bound adaptation (P3); readiness,
restart, cleanup, archive hashing, and rollback cover operations (P4).

## Immutable identity

| Item | Pinned value |
|---|---|
| Model | `Qwen/Qwen3.8-27B` |
| Model revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Checkpoint/artifact SHA-256 | `f5e9eb121362eac4d112889bcdbd53ac49d4a2a7a2fe6b1b9bad21c3efeaaf22` |
| Weight bytes | `55,563,006,776` (51.747 GiB across 18 safetensors shards) |
| Tokenizer SHA-256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Official template SHA-256 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Executed no-thinking template SHA-256 | `20158f9b80605efa9f0794b988d970861f5020d9f10b82bc65f1b38e0abf59bc` |
| Corpus SHA-256 | `531927d25f4e314fff8649007b96921af29d3e30809831f47eb43aeb7e80915d` |
| Runner source | fak `40d431b694ea4e39895cdc52b14560f8150b8bcf`; `internal/qwen38quantrun@r4+g28ed87a517` |
| Runtime | SGLang 0.5.13.post1; Torch 2.11.0+cu130; Transformers 5.8.1 |
| Image | `sha256:111ae1c2ea80cd2bd94b05a904bda6cf49f8e84e4a29f5dea14fc301a0e7d249` |
| Driver / CUDA | 580.173.02 / 13.0 |
| Hardware | 1× NVIDIA A100-SXM4-80GB, `sm_80` |
| Context / cache | 65,536 tokens / SGLang radix cache |
| Stale after | 2026-09-20 |

`provenance.json` lists all 32 checkpoint files. Its artifact digest is SHA-256 over the
path-sorted byte stream `path + NUL + file_sha256 + NUL + decimal_bytes + LF`; the test
recomputes that digest rather than trusting the recorded value.

The executed template is reproducible from the pinned official template by prefixing
these two LF-terminated lines exactly:

```jinja
{# issue-8308: frozen corpus budgets score visible answers, not hidden reasoning. #}
{%- set enable_thinking = false %}
```

The resulting file must hash to the executed-template value above. No reasoning parser
was enabled; the `qwen3_coder` tool parser remained enabled.

## Failure before, pass after

The first captured run is retained byte-for-byte in the `failure-before-*` files. It
used the official thinking-default template, the `qwen3` reasoning parser, and a 32,768
context limit. Only the three forced-tool trials passed; the other 15 trials either
exhausted their small answer budget in hidden reasoning or exceeded the launch context.
The report correctly stayed `HOLD`, while restart and cleanup still passed.

The corrected immutable launch disabled thinking through the hashed template, removed
the reasoning parser, and raised the served context to 65,536. The same checkpoint and
frozen corpus then passed 18/18 and produced `PROMOTE`. `witness_test.go` validates both
reports, both archive hashes, and the 3-pass/15-fail → 18-pass/0-fail transition.

## Quality and performance

| Workload | Passes | p50 latency |
|---|---:|---:|
| Text | 3/3 | 198.395 ms |
| JSON schema | 3/3 | 1,082.534 ms |
| Correlated tools | 3/3 | 1,139.019 ms |
| Coding reasoning | 3/3 | 162.699 ms |
| Long-context retrieval | 3/3 | 525.099 ms |
| Repeated workflow/cache | 3/3 | 431.996 ms |

| Resource / lifecycle metric | Result |
|---|---:|
| Load latency | 75,336 ms |
| Campaign completion throughput | 17.338910 tokens/s |
| Peak resident memory | 72,796,340,224 bytes (67.797 GiB) |
| Peak sampled power | 319.29 W |
| Cache cold latency | 431.996 ms |
| Cache warm latency | 304.715 ms |
| Warm-vs-cold latency delta | −29.463% |
| Warm-after-restart latency | 16,669.673 ms |
| Restart readiness / cleanup | PASS / PASS |
| Post-cleanup memory | 536 MiB |

SGLang reported zero cached-input tokens for both cold and warm requests. The latency
delta is therefore recorded, but this witness does **not** claim token-level cache reuse.

## Smaller-memory exclusion control

`smaller-topology-oom.json` loads the same artifact in BF16 with quantization, CPU
offload, and fallback all disabled, under an enforced 0.48 CUDA allocator fraction. The
39,322 MiB effective topology raised `CUDA_OUT_OF_MEMORY`, exited 1, never completed the
load, and was cleaned up. The physical board was an otherwise-idle A100-80GB; this is an
effective-memory ceiling control, not a claim that the run occurred on a physical 40GB
board. It preserves the smaller-topology OOM without co-scheduling on occupied hardware.

## Artifacts and reproduction

| Artifact | SHA-256 |
|---|---|
| `report.json` | `8baf30202b2d0d3594b80cc48402a19cfed2173815a49bc5e5f8ab563cc9ce7f` |
| `archive.json` | `ed057a1c492bb5117b8b1be578f2cf59a60d67cdb3fe0035bcf681bcfe35d4fe` |
| `summary.json` | `5e3ec92bdf1c1b8791646eb55dfdfc6b997a5a2c07e224151550452e3e4b724c` |
| `provenance.json` | `2bca830518006bbcd39758a215f18788af30f8bf9688240e89e984d90eb93d67` |
| `runtime.json` | `febfdaf56542d2ad8dc40e1410e85b8c7206f4b1e82c06bca7ba3ca82ce4b779` |
| `smaller-topology-oom.json` | `238210e53d43fae247ef85635119e2d3831fd71fc9b6c2ddb7bbf341e9e04682` |
| Failure-before report / archive | `cd13468b…` / `e2d156e4…` |

Build `cmd/qwen38campaign` from the pinned fak revision, download the immutable model
revision, verify the manifest and derived template hashes, and verify that
`docker image inspect config-sglang_service:latest --format '{{.Id}}'` equals the pinned
image ID. Then launch the model with the following secret-scrubbed command shape:

```text
docker run --name issue8308-bf16 --gpus device=2 --network host --ipc host \
  -v <artifact-f5e9eb12>:/model:ro \
  -v <template-20158f9b>:/chat_template.nothink.jinja:ro \
  config-sglang_service:latest \
  python3 -m sglang.launch_server --model-path /model \
  --served-model-name Qwen3.8-27B-BF16 --host 127.0.0.1 --port 18308 \
  --dtype bfloat16 --context-length 65536 --mem-fraction-static 0.80 \
  --chat-template /chat_template.nothink.jinja --tool-call-parser qwen3_coder \
  --api-key <ephemeral-secret>
```

Run the campaign with the checked-in corpus and the identity/environment values from
`report.json`; the private API key is intentionally absent from every artifact:

```text
qwen38campaign --config campaign-config.private.json \
  --corpus docs/benchmarks/qwen38-quant/corpus.json \
  --report report.json --archive archive.json
```

Independent local verification is:

```text
go test ./docs/_witnesses/issue-8308-qwen38-bf16
```

Rollback on any quality failure, identity/hash drift, active fallback, readiness failure,
or cleanup failure: stop and remove the BF16 container, retain the prior service, and
reject this reference until a fresh validator-clean campaign replaces it.
