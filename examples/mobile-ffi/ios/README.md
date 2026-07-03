# iOS Swift sample — gate an App Intent behind fak

An on-device agent proposes a tool call; this sample routes it through fak's
default-deny adjudicator floor over the FFI boundary **before** the call is
allowed to fire an Apple App Intent. A denied dangerous call (`send_sms`) never
dispatches; a benign one (`get_battery_level`) continues.

## Files

| File | Role |
| --- | --- |
| [`../libfakmobile.h`](../libfakmobile.h) | the C contract (`FakAdjudicate` / `FakFree`) |
| [`fak-Bridging-Header.h`](fak-Bridging-Header.h) | Swift ↔ C bridge (set as the target's bridging header) |
| [`AgentGate.swift`](AgentGate.swift) | Swift gate: adjudicate, then perform the App Intent iff allowed |

## Step 1 — cross-compile the fak archive for `ios/arm64`

Requires **macOS + Xcode** (the iOS SDK's clang). fak's release binary is
`CGO_ENABLED=0`, but the **FFI library form is a cgo `c-archive`**, so this build
sets `CGO_ENABLED=1` (see
[`../../../docs/integrations/mobile.md`](../../../docs/integrations/mobile.md)):

```sh
CGO_ENABLED=1 GOOS=ios GOARCH=arm64 \
  CC=$(xcrun --sdk iphoneos --find clang) \
  CGO_CFLAGS="-isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64 -miphoneos-version-min=15.0" \
  go build -buildmode=c-archive -o ios/libfakmobile_ios_arm64.a .
```

Run from the module root (`examples/mobile-ffi`). It emits the archive plus a
generated `libfakmobile_ios_arm64.h` whose two symbols match the checked-in
[`../libfakmobile.h`](../libfakmobile.h).

## Step 2 — add to the Xcode target

1. Drag `libfakmobile_ios_arm64.a` and `libfakmobile.h` into the app target.
2. Build Settings → **Objective-C Bridging Header** → `ios/fak-Bridging-Header.h`.
3. Call `AgentGate.performIfAllowed(...)` at the point you would fire an App
   Intent — it dispatches only on an `allow`.

## Witness note (operator input needed)

Unlike the Android `demo.c`, the iOS archive cannot be built on a non-macOS host.
The accepted device-free witness for this repo is a **macOS-host compile check**:
build the same archive for `GOARCH=arm64` against the macOS SDK
(`CC=$(xcrun --sdk macosx --find clang)`) and compile a tiny C `main` (the shape
of the Android [`demo.c`](../android/demo.c)) against it. This needs a
macOS/Xcode builder in CI — it is not reachable on the Linux/Windows trunk
builders. See the issue's open Witness question.

## Scope

A reference example, not a published app. No app-store artifact and no 32-bit
ARM — see the issue's Out-of-scope. The deny/allow decision is fak's; this
sample only carries it across the Swift/C seam.
