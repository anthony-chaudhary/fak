<!-- fak-metalgemm-key: zerocopy-stub-q4kweight-nondarwin-compile -->
# fix(metalgemm): restore non-Darwin zero-copy stub compilation

```routing
lane: internal/metalgemm
paths: ["internal/metalgemm/zerocopy_stub.go", "internal/metalgemm/zerocopy_stub_test.go", "docs/tickets/strix-performance/TICKET-metalgemm-zerocopy-stub-compile.md"]
expected_steps: 3
```

GitHub: [#12292](https://github.com/anthony-chaudhary/fak/issues/12292)

## Parent context

Corrective child of [#12243](https://github.com/anthony-chaudhary/fak/issues/12243), which landed the zero-copy Darwin implementation and complementary stub.

## Current state

`q4k.go` defines `Q4KWeight` only for `darwin && arm64 && cgo`, while `zerocopy_stub.go` is selected outside that constraint and refers to `Q4KWeight` at lines 57 and 62. Windows packages that transitively import `internal/metalgemm` fail to compile.

## Working spine

Non-Darwin build -> inert zero-copy API -> compile-safe `ErrMetalUnavailable` result -> existing callers retain fallback behavior.

## Core through-line

Correct only the non-Darwin stub/type boundary and add a focused compile regression without enabling Metal or weakening Darwin ownership of the real Q4_K implementation.

## Gold-plating boundary

No Metal kernel, GGUF loader, Q4_K math, Darwin implementation, rawdecode, modelbench, hardware, benchmark, performance, Windows emulation, or external-runtime fallback.

## Done condition / witness

`go test ./internal/rawdecode ./cmd/modelbench -count=1` compiles past the Metal stub on Windows and the focused stub test preserves `ErrMetalUnavailable`.

## Done condition

- [x] Non-Darwin compilation succeeds.
- [x] Stub zero-copy construction remains unavailable.
- [x] The stub never claims a shared Metal buffer.
- [x] Darwin behavior and build constraints remain unchanged.

## Verifiable Witness

The Windows/amd64 device-free witness passes:

```text
go test ./internal/metalgemm -count=1
go test ./internal/rawdecode ./cmd/modelbench -count=1
go vet ./internal/metalgemm ./internal/rawdecode ./cmd/modelbench
```

Darwin and physical Metal checks remain unrun.

## Likely files

- `internal/metalgemm/zerocopy_stub.go`
- `internal/metalgemm/zerocopy_stub_test.go`
- this ticket

## Lane

`internal/metalgemm`; one package, three expected steps.

## Blast radius and fallback

The regression blocks Windows/Linux consumers before runtime. Until corrected, the only safe fallback is the prior green revision; no runtime fallback can execute.

## Closure binding

The resolving commit cites #12292 and carries `(fak metalgemm)`.
