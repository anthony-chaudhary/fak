# Android NDK sample — gate an Intent behind fak

An on-device agent proposes a tool call; this sample routes it through fak's
default-deny adjudicator floor over the FFI boundary **before** the call is
allowed to become an `android.content.Intent`. A denied dangerous call
(`send_sms`) never reaches `startActivity`; a benign one (`get_battery_level`)
continues.

## Files

| File | Role |
| --- | --- |
| [`../libfakmobile.h`](../libfakmobile.h) | the C contract (`FakAdjudicate` / `FakFree`) |
| [`fak_gate.c`](fak_gate.c) | JNI bridge: `AgentGate.nativeAdjudicate(String) -> String` |
| [`AgentGate.kt`](AgentGate.kt) | Kotlin gate: adjudicate, then `startActivity` iff allowed |
| [`demo.c`](demo.c) | device-free witness: links the archive, drives deny/allow |

## Step 1 — cross-compile the fak archive for `android/arm64`

Requires the Android NDK (`$ANDROID_NDK_HOME`). fak's release binary is
`CGO_ENABLED=0`, but the **FFI library form is a cgo `c-archive`**, so this build
sets `CGO_ENABLED=1` and points `CC` at the NDK's clang (see
[`../../../docs/integrations/mobile.md`](../../../docs/integrations/mobile.md)
for why the two build modes differ):

```sh
NDK_BIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin
CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
  CC=$NDK_BIN/aarch64-linux-android21-clang \
  go build -buildmode=c-archive -o android/libfakmobile_android_arm64.a .
```

Run this from the module root (`examples/mobile-ffi`). It emits the archive plus
a generated `libfakmobile_android_arm64.h` (its two symbols match the checked-in
[`../libfakmobile.h`](../libfakmobile.h)).

## Step 2a — device-free witness (`demo.c`)

Link the standalone witness and run it under an arm64 executor (emulator shell,
`adb push`, or `qemu-aarch64`):

```sh
$NDK_BIN/aarch64-linux-android21-clang \
  -I . android/demo.c android/libfakmobile_android_arm64.a -o android/demo
# then run android/demo on an arm64 target; expected tail:
#   android demo: OK — dangerous denied, benign continued, unknown failed closed
```

## Step 2b — in the app (JNI)

Compile `fak_gate.c` into the app's native library (name it `fakgate` so
`System.loadLibrary("fakgate")` in `AgentGate.kt` resolves), linking the archive.
A minimal `CMakeLists.txt` for the module's `externalNativeBuild`:

```cmake
add_library(fakgate SHARED fak_gate.c)
target_include_directories(fakgate PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(fakgate ${CMAKE_SOURCE_DIR}/libfakmobile_android_arm64.a)
find_library(log-lib log)
target_link_libraries(fakgate ${log-lib})
```

Then `AgentGate.dispatchIfAllowed(...)` is the enforcement point: the Intent is
started only on an `allow`.

## Scope

A reference example, not a published app. No app-store artifact and no 32-bit
ARM (armv7) — see the issue's Out-of-scope. The deny/allow decision is fak's;
this sample only carries it across the JNI seam.
