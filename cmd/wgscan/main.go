package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func main() { os.Exit(run(".", os.Stdout)) }

func run(root string, stdout io.Writer) int {
	rep, err := windowgate.ScanTree(root)
	if err != nil {
		fmt.Fprintln(stdout, "ERR", err)
		return 0
	}
	dump := func(name string, xs []string) {
		fmt.Fprintf(stdout, "== %s (%d) ==\n", name, len(xs))
		for _, x := range xs {
			fmt.Fprintln(stdout, "  ", x)
		}
	}
	dump("PSInstallers", rep.PSInstallers)
	dump("PSStartProcesses", rep.PSStartProcesses)
	dump("PySpawns", rep.PySpawns)
	dump("PyCandidates", rep.PyCandidates)
	dump("GoExecs", rep.GoExecs)
	dump("GoCandidates", rep.GoCandidates)
	fmt.Fprintln(stdout, "OK:", rep.OK())
	return 0
}
