package main

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func main() {
	rep, err := windowgate.ScanTree(".")
	if err != nil {
		fmt.Println("ERR", err)
		return
	}
	dump := func(name string, xs []string) {
		fmt.Printf("== %s (%d) ==\n", name, len(xs))
		for _, x := range xs {
			fmt.Println("  ", x)
		}
	}
	dump("PSInstallers", rep.PSInstallers)
	dump("PSStartProcesses", rep.PSStartProcesses)
	dump("PySpawns", rep.PySpawns)
	dump("PyCandidates", rep.PyCandidates)
	dump("GoExecs", rep.GoExecs)
	dump("GoCandidates", rep.GoCandidates)
	fmt.Println("OK:", rep.OK())
}
