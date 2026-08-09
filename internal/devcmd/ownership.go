package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const runtimeImportRoot = "github.com/anthony-chaudhary/fak/cmd/fak"

// RunOwnership emits the complete runtime/dev ownership and dependency witness.
// It is the first command implementation migrated out of package main.
func RunOwnership(stdout, stderr io.Writer, root string, asJSON bool) int {
	report, err := devindex.BuildOwnershipReport(root, "./cmd/fak", runtimeImportRoot)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index ownership: %v\n", err)
		return 1
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak-dev index ownership: encode json: %v\n", err)
			return 1
		}
		return 0
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
