// fak — the Fused Agent Kernel. One Go binary: tool-loop harness + in-process
// adjudication + tool vDSO + pre-flight ladder + context-MMU, driving any
// OpenAI-compatible engine.
//
// The module is the repository root, so it installs directly:
//   go install github.com/anthony-chaudhary/fak/cmd/fak@latest
// The entire external dependency set is two golang.org/x extended-standard-library
// modules — golang.org/x/term (the CLI's terminal probe and size/raw-mode handling)
// and golang.org/x/sys indirectly through it — pinned by a 4-line go.sum.
module github.com/anthony-chaudhary/fak

go 1.26

// Pin the patched toolchain: go1.26.6 fixes the reachable standard-library
// vulnerabilities reported by govulncheck against go1.26.5.
// GOTOOLCHAIN=auto fetches the same patched release in CI and locally.
toolchain go1.26.6

require golang.org/x/term v0.44.0

require golang.org/x/sys v0.46.0 // indirect
