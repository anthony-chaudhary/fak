# extdriver — a worked out-of-tree fak driver

This is the worked example proving fak's ABI is importable by an **external
module**. It claims both vendor extension numbers end-to-end and registers one
real driver, importing only the public `pkg/abi` surface.

## Why it exists

fak's frozen spine lives in `internal/abi`. Go's `internal/` rule seals that
package, so no module outside `github.com/anthony-chaudhary/fak` can import it.
`pkg/abi` is the importable re-export of the **vendor/driver-facing** subset of
that spine (a thin set of type aliases + `var`/`const` re-exports — same
underlying types, zero added behavior). A driver implementing `pkgabi.Adjudicator`
already satisfies `internalabi.Adjudicator` because they are the same type.

A driver author imports `pkg/abi`, **never** `internal/abi`.

## What it does

`main.go` imports `github.com/anthony-chaudhary/fak/pkg/abi` and:

1. **Claims an `OpCode` in the `OpsVendor` range** `[1<<16, 1<<17)` via
   `abi.RegisterOp`. `RegisterOp` panics on a clash, so a clean registration is
   the proof the number is yours.
2. **Claims a `VerdictKind` in the `VerdictsVendor` range** `[1024, 1<<16)` via
   `abi.RegisterVerdictKind`, with a fold-rank and a fail-closed `FallbackDeny`.
   `abi.FoldRank(kind)` then resolves it from the frozen lattice — confirming the
   registration landed.
3. **Registers a real `Adjudicator`** that denies a named tool (`refund_payment`)
   with the vendor verdict kind and the core reason `POLICY_BLOCK`, and `Defer`s
   on every other tool (the fold identity).

`main` exercises the round-trip and exits **non-zero** if any leg fails, so
running it is a witness — not just a compile.

## Build and run

This is a **separate Go module** (its own `go.mod`). The root module's
`go build ./...` / `go test ./...` do **not** descend into it, so build it
explicitly:

```sh
cd examples/extdriver
go build ./...
go run .
```

The build-and-run **completes in a few seconds** and is **deterministic** — the same
output every run, with no model, network, or randomness in the loop.

## Expected output

The end-to-end claim proof (a full captured run is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)):

```
fak out-of-tree driver — ABI importable via pkg/abi
  ABI version: v0.1

[1] claimed OpCode 65578 (OpsVendor range [65536,131072))
    registered without a clash -> opcode is claimed

[2] claimed VerdictKind 1031 (VerdictsVendor range [1024,65536))
    FoldRank(1031) = 200 -> kind is registered in the frozen lattice

[3] Adjudicate(tool="refund_payment") -> Kind=1031 Reason=POLICY_BLOCK By=extdriver/denyTool
    Adjudicate(tool="read_file") -> Kind=5 (VerdictDefer) — fold identity holds

extdriver: OK — both vendor numbers claimed and the driver round-trip passed
```

The `go.mod` here uses a `replace github.com/anthony-chaudhary/fak => ../..` so
the example builds against this checkout. A real consumer would instead
`require` a tagged fak version.

## Scope — what this does *not* claim

This is a **witness of the ABI seam**, not a benchmark or a kernel demo. It does **not**
claim any performance result, does **not** exercise the gateway, serving, or the in-kernel
model, and does **not** prove that the *core* `internal/abi` is importable (it can't be —
that's the whole point; only the `pkg/abi` re-export is). It proves exactly one thing: an
external module can claim both vendor number ranges and register a working `Adjudicator`
through the public surface. The honesty ledger for the underlying capability is
[`../../CLAIMS.md`](../../CLAIMS.md).

## CI note

Because this is a nested module, the root CI's single-module `go build ./...` /
`go test ./...` will not build it. To keep it from rotting, it needs its own CI
step (`cd examples/extdriver && go build ./... && go run .`). See issue #454.
