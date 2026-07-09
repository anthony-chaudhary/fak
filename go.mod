// fak — the Fused Agent Kernel. One Go binary: tool-loop harness + in-process
// adjudication + tool vDSO + pre-flight ladder + context-MMU, driving any
// OpenAI-compatible engine.
//
// The module is the repository root, so it installs directly:
//   go install github.com/anthony-chaudhary/fak/cmd/fak@latest
// Zero external dependencies (standard library only) — there is no go.sum.
module github.com/anthony-chaudhary/fak

go 1.26

require golang.org/x/term v0.44.0

require golang.org/x/sys v0.46.0 // indirect
