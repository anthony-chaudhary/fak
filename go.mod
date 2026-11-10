// fak — the Fused Agent Kernel. One Go binary: tool-loop harness + in-process
// adjudication + tool vDSO + pre-flight ladder + context-MMU, driving any
// OpenAI-compatible engine.
//
// The module is the repository root, so it installs directly:
//   go install github.com/anthony-chaudhary/fak/cmd/fak@latest
// The entire external dependency set is four pinned golang.org/x
// extended-standard-library modules — x/image, x/sys, x/term, and x/text —
// pinned by an 8-line go.sum.
module github.com/anthony-chaudhary/fak

go 1.26.0

// Pin the patched toolchain: go1.26.10 fixes GO-2026-6809 in crypto/x509,
// in addition to the earlier standard-library security fixes.
// GOTOOLCHAIN=auto selects this release when the default toolchain is older.
toolchain go1.26.10

require golang.org/x/term v0.47.0

require golang.org/x/sys v0.49.0

require (
	golang.org/x/image v0.47.0
	golang.org/x/text v0.43.0 // indirect
)
