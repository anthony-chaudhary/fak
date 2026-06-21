//go:build amd64

package model

// fmaCrossPathTol bounds the divergence allowed between two algebraically-identical f32 code
// paths in the "bit-exact" rungs (the reposition invariant, the profiler twin). On amd64 the
// gc compiler does NOT auto-fuse multiply-add (FMA3 isn't universal across amd64 parts), so
// two equivalent Go expressions round identically and these rungs are EXACTLY 0 — keep them
// byte-exact here. The non-amd64 twin of this constant (fmatol_other_test.go) explains why
// they cannot be 0 on arches with mandatory hardware FMA.
const fmaCrossPathTol = 0.0
