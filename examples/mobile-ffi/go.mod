// mobile-ffi is a SEPARATE Go module (not part of the root module's ./...) that
// proves fak's default-deny adjudicator floor can be reached across an FFI
// boundary — the same seam an on-device agent on Android (NDK) or iOS (Swift)
// crosses BEFORE a proposed tool call becomes an Android Intent / Apple App
// Intent. It imports ONLY the public pkg/abi surface (Go's internal/ rule seals
// internal/abi from an out-of-tree module), exactly like ../extdriver.
//
// The replace directive points fak at this checkout so the example builds
// against local source; a real consumer would `require` a tagged version.
module github.com/anthony-chaudhary/fak/examples/mobile-ffi

go 1.26

require github.com/anthony-chaudhary/fak v0.0.0

replace github.com/anthony-chaudhary/fak => ../..
