<!-- fak-qwen38-key: parsed-gguf-tokenizer-template-identity-seam -->
# feat(qwen38quantrun): expose the parsed GGUF identity seam

```routing
lane: qwen38-prompt-packet
paths: ["internal/qwen38quantrun/prompt_packet_gguf.go", "internal/qwen38quantrun/prompt_packet_test.go", "docs/tickets/strix-performance/TICKET-qwen38-parsed-gguf-identity-seam.md"]
expected_steps: 4
```

GitHub: [#12287](https://github.com/anthony-chaudhary/fak/issues/12287)

## Parent context

Parent [#9883](https://github.com/anthony-chaudhary/fak/issues/9883); dependency for [#12096](https://github.com/anthony-chaudhary/fak/issues/12096) and [#5881](https://github.com/anthony-chaudhary/fak/issues/5881).

## Current state

The strict Qwen3.8 GGUF tokenizer/template identity implementation already accepts an in-memory `*ggufload.File`, but that seam is private. Other loaders would have to reopen a path or duplicate canonicalization.

## Working spine

Caller-owned exact GGUF stream -> caller-parsed header -> existing strict canonical identity -> paired tokenizer/template digests or typed refusal.

## Core through-line

Export the existing parsed-header function; keep the path helper as a compatible wrapper; prove tokenizer and template mutations remain isolated and missing metadata still fails closed.

## Gold-plating boundary

No parser, tokenizer execution, prompt rendering, rawdecode, modelbench, hardware, service, model download, second hash algorithm, filename inference, or sidecar metadata.

## Done condition / witness

The exported same-header function and its wrapper pass `go test ./internal/qwen38quantrun -run TestDerivePromptPacketGGUFIdentity -count=1` and `go vet ./internal/qwen38quantrun`.

## Done condition

- [ ] Path helper delegates to the exported parsed-header seam.
- [ ] Tokenizer mutation changes only its digest.
- [ ] Template mutation changes only its digest.
- [ ] Missing embedded metadata still returns an error.

## Verifiable Witness

The focused commands above are device-free and deterministic. No physical or performance claim is made.

## Likely files

- `internal/qwen38quantrun/prompt_packet_gguf.go`
- `internal/qwen38quantrun/prompt_packet_test.go`
- this ticket

## Lane

`qwen38-prompt-packet`; one Go package, four expected steps.

## Closure binding

The resolving commit cites #12287 and carries `(fak qwen38quantrun)`.
