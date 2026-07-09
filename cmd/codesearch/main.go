// Command codesearch is a standalone front door to the epic #3434
// code-intelligence engine (internal/codesearch): trigram regex/literal search,
// AST shape queries, call-graph traversal, and RRF-fused feature retrieval over a
// Go tree. It exists as its own binary so the engine is user-runnable independent
// of the main `fak` binary; the same engine also backs the `fak dev codesearch`
// verb. Run `go run ./cmd/codesearch <sub> ...` — see the usage banner for subs.
package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/codesearch"
)

func main() {
	os.Exit(codesearch.Run(os.Args[1:], os.Stdout, os.Stderr))
}
