# mobile-ffi — route on-device tool calls through fak before an intent fires

The reference example for epic #633 segment A ("the enforcement layer
Apple/Google leave empty"), issue #1042. An on-device agent proposes a tool
call; it is routed through fak's **default-deny adjudicator floor** across an FFI
boundary **before** it is allowed to become an **Android Intent** or an **Apple
App Intent**. A denied dangerous call never reaches the dispatcher; a benign one
continues — least-privilege-per-tool as a gate, not developer discipline.

This is a **reference example, not a product** (no app-store artifact, no armv7).
It requires **no kernel change**: the work is the FFI boundary plus two minimal
samples.

## Layout

```
mobile-ffi/
  adjudicate.go       pure-Go core: the floor (on pkg/abi) + Decide(json) Decision
  main.go             go-run witness of the deny/allow/fail-closed round-trip
  adjudicate_test.go  table test of the same round-trip (CGO_ENABLED=0)
  libfakmobile.go     the cgo //export shim (built only with -buildmode=c-archive)
  libfakmobile.h      the stable hand-authored C contract the samples include
  android/            Android NDK sample (JNI bridge + Kotlin gate + device-free demo.c)
  ios/                iOS Swift sample (bridging header + Swift gate)
```

## The FFI contract

One call, JSON in and JSON out (full surface:
[`../../docs/integrations/mobile.md`](../../docs/integrations/mobile.md)):

```c
char *FakAdjudicate(char *toolCallJSON);   // {"tool":"send_sms",...} -> Decision JSON
void  FakFree(char *p);                    // release the returned buffer
```

The host dispatches the intent **iff** the returned Decision's `"allow"` is true.

## This is a separate Go module

Like [`../extdriver`](../extdriver), `mobile-ffi` has its own `go.mod` and
imports only the public `pkg/abi` surface, so the root module's `go build ./...`
/ `go test ./...` do **not** descend into it (the root CI stays green regardless).
Build and witness the core on any host (no cgo, no device):

```sh
go -C examples/mobile-ffi test ./...
go -C examples/mobile-ffi run .
```

Expected `run` output is captured in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md); the
final `mobile-ffi: OK` line and a zero exit are the witness.

## Build the arm64 FFI archive

fak's release binary is `CGO_ENABLED=0`, but a C-callable library is a cgo
`c-archive` (`CGO_ENABLED=1` + a cross C compiler). Per-platform recipes:
[android/README.md](android/README.md), [ios/README.md](ios/README.md). Why the
two build modes differ, and what is/ isn't witnessed on the trunk builders, is in
[`../../docs/integrations/mobile.md`](../../docs/integrations/mobile.md).

## Scope — what this does not claim

It does not ship a store artifact, does not build the on-device UI, and does not
fold in the phone `EngineDriver` ticket (#633 Tier 2 item 1) — the floor here is
engine-free by construction. It proves exactly one thing: fak's default-deny
verdict is reachable across the Android/iOS FFI seam and correctly denies a
dangerous proposed call while continuing a benign one.
