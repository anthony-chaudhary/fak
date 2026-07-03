# Mobile FFI — gating on-device tool calls through fak

fak's default-deny adjudicator floor is reachable from an on-device agent on
**Android (NDK)** and **iOS (Swift)** across a small C FFI boundary, so a
proposed tool call passes least-privilege-per-tool enforcement **before** it
becomes an Android `Intent` or an Apple App Intent. This is the concrete
delivery of epic #633 segment A — "the enforcement layer Apple/Google leave
empty": Android auto-authorizes "related sub-tools" from one coarse grant (it
names the risk "excessive agency"); Apple's intent resolver trusts its own
dispatch. fak makes the call pass its floor first.

The worked reference example is [`examples/mobile-ffi`](../../examples/mobile-ffi/).
This document is the FFI surface and the build contract.

## The surface

One adjudication call plus its buffer-release companion (C contract in
[`examples/mobile-ffi/libfakmobile.h`](../../examples/mobile-ffi/libfakmobile.h)):

```c
char *FakAdjudicate(char *toolCallJSON);
void  FakFree(char *p);
```

- **In:** the proposed tool call as a JSON C string, `{"tool":"<name>","args":{…}}`.
  Only `tool` drives the reference floor's verdict; `args` are carried for the
  host's own logging.
- **Out:** a JSON `Decision` C string:

  ```json
  {"tool":"send_sms","allow":false,"verdict":"DENY","reason":"POLICY_BLOCK","by":"mobile/floor"}
  ```

  The host dispatches the intent **iff** `allow` is `true`. `verdict` is one of
  `ALLOW` / `DENY` / `DEFAULT_DENY`; `reason` is a closed `ReasonCode` name
  (e.g. `POLICY_BLOCK`, `MALFORMED`, `DEFAULT_DENY`).
- **Ownership:** the returned `char *` is malloc'd by fak and owned by the
  caller — release it once with `FakFree` (`FakFree(NULL)` is a no-op).
- **Purity:** both calls are thread-safe and side-effect-free; the reference
  floor holds an immutable policy, so the same input always yields the same
  Decision. No engine, network, or model is in the loop (the deny/allow demo
  needs no on-device model — it pairs with, but does not depend on, the phone
  `EngineDriver` ticket).

The floor mirrors fak's in-tree reference monitor via the public `pkg/abi`
`Adjudicator` seam: a named dangerous tool is a provable `DENY(POLICY_BLOCK)`, a
read-shaped benign tool (`get_`/`read_`/`list_`/`query_`) is `ALLOW`, and
anything else `DEFAULT_DENY`s (fail-closed). A production host would load a
policy manifest instead of the pinned reference set.

## The two build modes — reconciled

The issue premise cites fak's `linux/arm64` release, which is
**`CGO_ENABLED=0`** (a static Go binary, nothing to port — the same NEON Q8 path
Apple silicon ships). A C-callable **library**, however, is a different artifact:
`go build -buildmode=c-archive` **requires cgo** (`CGO_ENABLED=1`) and a C
cross-compiler for the target. Both are true at once:

| Artifact | Build mode | CGO | Toolchain |
| --- | --- | --- | --- |
| Release CLI (`fak`) | default binary | `0` | Go only |
| Mobile FFI archive | `c-archive` | `1` | NDK clang (Android) / Xcode clang (iOS) |

The Go source is portable across both; only the **link form** differs. The cgo
shim ([`libfakmobile.go`](../../examples/mobile-ffi/libfakmobile.go)) is guarded
by `//go:build cgo`, so under the repo's `CGO_ENABLED=0` build it is excluded and
the pure-Go core still compiles, vets, and tests — the C boundary adds a symbol,
never a decision.

### android/arm64

```sh
NDK_BIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin
CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
  CC=$NDK_BIN/aarch64-linux-android21-clang \
  go build -buildmode=c-archive -o libfakmobile_android_arm64.a .
```

### ios/arm64

```sh
CGO_ENABLED=1 GOOS=ios GOARCH=arm64 \
  CC=$(xcrun --sdk iphoneos --find clang) \
  CGO_CFLAGS="-isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64 -miphoneos-version-min=15.0" \
  go build -buildmode=c-archive -o libfakmobile_ios_arm64.a .
```

Each build also emits a generated `libfakmobile_<goos>_<goarch>.h`; its two
symbols match the hand-authored `libfakmobile.h` the samples include.

## What is witnessed, and what needs operator hardware

- **Witnessed on any host (CGO_ENABLED=0, no device):** the pure-Go core round
  trip — `go -C examples/mobile-ffi test ./...` and `go -C examples/mobile-ffi
  run .` (deny dangerous, continue benign, fail-closed unknown). This is the
  same decision the C shim carries.
- **Needs the Android NDK:** the `android/arm64` `c-archive` cross-compile and
  the device-free `android/demo.c` link+run under an arm64 executor.
- **Needs macOS + Xcode:** the `ios/arm64` `c-archive` build. It cannot be
  produced on the Linux/Windows trunk builders. The proposed device-free witness
  is a macOS-host compile check (build for `GOARCH=arm64` against the macOS SDK
  and link a tiny C `main`); wiring a macOS builder into CI is the open operator
  question tracked on issue #1042.

## Promotion / demotion / invalidation (gen/next)

- **Promotion evidence** (moves this from `gen/next` toward `now`): a CI job on a
  macOS + NDK builder that cross-compiles both `c-archive` archives and runs the
  device-free `demo.c` witness green — closing the two hardware-gated legs above.
- **Demotion / retirement evidence:** if the phone `EngineDriver` lands an
  end-to-end on-device demo that already exercises this floor, this standalone
  sample is redundant and can be folded into that demo.
- **Invalidating assumption:** that `-buildmode=c-archive` is the FFI form the
  mobile hosts want. If a target instead needs a `c-shared` `.so`/`.dylib` or an
  XCFramework, the archive step (not the surface) changes; the `FakAdjudicate` /
  `FakFree` contract and the pure-Go core are unaffected.
