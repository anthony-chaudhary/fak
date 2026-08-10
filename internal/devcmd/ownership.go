package devcmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const runtimeImportRoot = "github.com/anthony-chaudhary/fak/cmd/fak"

// runtimeClosurePattern is the go-list pattern whose dependency closure IS the
// runtime artifact: exactly what linking cmd/fak pulls in. It must never widen to
// "./..." — that closure is the whole module (every dev-only and test-only package
// too), so the reported package/internal counts would describe no shipped binary
// (#6022). The leak BFS is rooted at runtimeImportRoot and so is insensitive to the
// widening; the COUNTS are not.
const runtimeClosurePattern = "./cmd/fak"

// RunOwnership emits the complete runtime/dev ownership and dependency witness.
// It is the first command implementation migrated out of package main, and the
// single body behind `fak-dev index ownership`.
func RunOwnership(stdout, stderr io.Writer, root string, asJSON bool) int {
	report, err := devindex.BuildOwnershipReport(root, runtimeClosurePattern, runtimeImportRoot)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index ownership: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak-dev index ownership")
	}
	var runtime, dev, shared int
	for _, command := range report.Commands {
		switch command.Owner {
		case devindex.OwnerRuntime:
			runtime++
		case devindex.OwnerDev:
			dev++
		case devindex.OwnerShared:
			shared++
		}
	}
	fmt.Fprintf(stdout, "command ownership: runtime=%d dev=%d shared=%d total=%d\n", runtime, dev, shared, len(report.Commands))
	fmt.Fprintf(stdout, "runtime graph: packages=%d internal=%d dev-leaks=%d\n", report.Graph.PackageCount, report.Graph.InternalCount, len(report.Graph.Leaks))
	for _, leak := range report.Graph.Leaks {
		fmt.Fprintf(stdout, "LEAK %s: %s\n", leak.Forbidden, strings.Join(leak.Path, " -> "))
	}
	return 0
}
