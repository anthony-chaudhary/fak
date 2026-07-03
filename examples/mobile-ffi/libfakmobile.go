//go:build cgo

// libfakmobile.go is the C-callable boundary — the ONLY cgo file in this module.
// It is built exclusively under `-buildmode=c-archive` (which sets CGO_ENABLED=1
// and requires a C cross-compiler: the Android NDK's clang for android/arm64,
// Xcode's clang for ios/arm64). Under the repo's CGO_ENABLED=0 build the
// `//go:build cgo` constraint excludes this file entirely, so `go build`,
// `go vet`, and `go test` in this module compile only the pure-Go core — the C
// shim never widens or narrows a decision, it only re-exposes Decide over a
// C-string boundary.
//
// Producing the archive + header (see docs/integrations/mobile.md for the full
// per-platform recipe):
//
//	# android/arm64 (needs $ANDROID_NDK_HOME)
//	CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
//	  CC=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android21-clang \
//	  go build -buildmode=c-archive -o libfakmobile_android_arm64.a .
//
//	# ios/arm64 (needs macOS + Xcode)
//	CGO_ENABLED=1 GOOS=ios GOARCH=arm64 CC=$(xcrun --sdk iphoneos --find clang) \
//	  CGO_CFLAGS="-isysroot $(xcrun --sdk iphoneos --show-sdk-path) -arch arm64" \
//	  go build -buildmode=c-archive -o libfakmobile_ios_arm64.a .
//
// Each build emits the archive plus a generated `libfakmobile.h` declaring the
// two symbols below. A committed reference copy of that header lives alongside
// this file as libfakmobile.h so the platform samples have something to compile
// against before the archive is built.
package main

/*
#include <stdlib.h>
*/
import "C"

import "unsafe"

// FakAdjudicate is the on-device gate. The host passes the proposed tool call as
// a JSON C string (`{"tool":"send_sms","args":{...}}`) and receives a JSON
// Decision C string (`{"tool":"send_sms","allow":false,"verdict":"DENY",...}`).
// The host dispatches the Android Intent / Apple App Intent IFF `.allow` is true.
//
// The returned pointer is heap-allocated by Go's C.CString (malloc); the caller
// OWNS it and MUST release it with FakFree to avoid a leak. Never free it with a
// mismatched allocator.
//
//export FakAdjudicate
func FakAdjudicate(toolCallJSON *C.char) *C.char {
	d := Decide(C.GoString(toolCallJSON))
	return C.CString(d.marshalJSON())
}

// FakFree releases a string returned by FakAdjudicate. Passing NULL is a no-op.
//
//export FakFree
func FakFree(p *C.char) {
	if p != nil {
		C.free(unsafe.Pointer(p))
	}
}
