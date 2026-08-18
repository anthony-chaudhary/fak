package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessmix"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func runHarnessMix(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness mix", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var imports harnessLayerFlag
	fs.Var(&imports, "import", "verified mix-ready harness lock (repeatable)")
	output := fs.String("output", "", "write mixed product lock")
	receipt := fs.String("receipt", "", "write mix receipt (default: <output>.mix.json)")
	contextBudget := fs.Int("context-budget", 0, "maximum deduplicated context tokens")
	memoryBudget := fs.Int("memory-budget-mib", 0, "maximum deduplicated memory MiB")
	workerBudget := fs.Int("worker-budget", 0, "maximum deduplicated workers")
	jsonView := fs.Bool("json", false, "emit lock and receipt")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	paths := imports.values()
	if len(paths) < 2 || *output == "" {
		fmt.Fprintln(stderr, "fak harness mix: repeat --import at least twice and pass --output")
		return 2
	}
	locks := make([]harnessresolve.Lock, 0, len(paths))
	for _, path := range paths {
		lock, err := readHarnessPreviewLock(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness mix: import %s: %v\n", path, err)
			return 1
		}
		locks = append(locks, *lock)
	}
	result, err := harnessmix.Mix(locks, harnessmix.Limits{ContextTokens: *contextBudget, MemoryMiB: *memoryBudget, Workers: *workerBudget})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness mix: %v\n", err)
		return 1
	}
	if *receipt == "" {
		*receipt = *output + ".mix.json"
	}
	if err = writeDerivedJSON(*output, result.Lock); err != nil {
		fmt.Fprintf(stderr, "fak harness mix: output: %v\n", err)
		return 1
	}
	if err = writeDerivedJSON(*receipt, result.Receipt); err != nil {
		fmt.Fprintf(stderr, "fak harness mix: receipt: %v\n", err)
		return 1
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err = enc.Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "HARNESS MIX | VERIFIED\nimports: %d\nresult: %s\ncomponents: %d | deduplicated %d\nbudget: %d context tokens | %d MiB | %d workers\nwritten: %s\nreceipt: %s\n", len(paths), result.Lock.ID, len(result.Lock.Components), len(result.Receipt.Deduplicated), result.Lock.Budget.ContextTokens, result.Lock.Budget.MemoryMiB, result.Lock.Budget.Workers, *output, *receipt)
	fmt.Fprintf(stdout, "next: fak harness inspect --lock %s\n", *output)
	return 0
}

var _ = os.ErrNotExist
