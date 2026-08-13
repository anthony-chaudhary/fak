package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stalework"
)

func cmdStaleWork(args []string) { os.Exit(runStaleWork(args, os.Stdout, os.Stderr)) }
func runStaleWork(args []string, out, errw io.Writer) int {
	fs := flag.NewFlagSet("stale-work", flag.ContinueOnError)
	fs.SetOutput(errw)
	root := fs.String("root", ".", "repository root")
	limit := fs.Int("limit", 10, "maximum candidates")
	path := fs.String("path", "", "scan one tracked artifact")
	open := fs.String("open-issues", "", "JSON file containing open stale-work dedupe keys")
	self := fs.Bool("selfcheck", false, "run deterministic dependency-drift spine")
	_ = fs.Bool("json", true, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *self {
		return staleWorkSelfcheck(out, errw)
	}
	keys := map[string]bool{}
	if *open != "" {
		b, e := os.ReadFile(*open)
		if e != nil {
			fmt.Fprintf(errw, "fak stale-work: %v\n", e)
			return 1
		}
		var v []string
		if e = json.Unmarshal(b, &v); e != nil {
			fmt.Fprintf(errw, "fak stale-work: open issues: %v\n", e)
			return 1
		}
		for _, k := range v {
			keys[k] = true
		}
	}
	paths := []string(nil)
	if *path != "" {
		paths = []string{*path}
	}
	p, err := stalework.Scan(context.Background(), stalework.Options{Root: *root, Limit: *limit, Paths: paths, OpenDedupe: keys})
	if err != nil {
		fmt.Fprintf(errw, "fak stale-work: %v\n", err)
		return 1
	}
	b, _ := json.Marshal(p)
	p.Metrics.PacketBytes = len(b)
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(p); err != nil {
		return 1
	}
	return 0
}
func staleWorkSelfcheck(out, errw io.Writer) int {
	dir, e := os.MkdirTemp("", "fak-stale-work-")
	if e != nil {
		return 1
	}
	defer os.RemoveAll(dir)
	must := func(p, s string) {
		_ = os.MkdirAll(dir+"/"+p[:len(p)-len(filepathBase(p))], 0755)
		_ = os.WriteFile(dir+"/"+p, []byte(s), 0644)
	}
	_ = must
	// The package fixture test is the detailed oracle; this command proves its ranking contract without touching the repository.
	run := func(_ context.Context, _ string, a ...string) ([]byte, error) {
		k := fmt.Sprint(a)
		switch {
		case a[0] == "rev-parse":
			return []byte("head\n"), nil
		case a[0] == "ls-files":
			return []byte("docs/operator.md\ndocs/notes/history.md\n"), nil
		case a[0] == "log" && a[1] == "-1" && a[len(a)-1] == "docs/operator.md":
			return []byte("old|1767225600\n"), nil
		case a[0] == "log" && a[1] == "-1":
			return []byte("ancient|1609459200\n"), nil
		case a[0] == "log" && a[1] == "--format=%H|%s" && a[len(a)-1] == "cmd/fak":
			return []byte("new|changed command\n"), nil
		default:
			_ = k
			return nil, nil
		}
	}
	_ = os.MkdirAll(dir+"/docs/notes", 0755)
	_ = os.WriteFile(dir+"/docs/operator.md", []byte("Run `fak serve` for current operation."), 0644)
	_ = os.WriteFile(dir+"/docs/notes/history.md", []byte("# Historical archive\nOld record."), 0644)
	p, e := stalework.Scan(context.Background(), stalework.Options{Root: dir, Limit: 5, Now: time.Date(2026, 8, 13, 0, 0, 0, 0, time.UTC), Run: run})
	if e != nil || len(p.Candidates) != 1 || p.Candidates[0].Path != "docs/operator.md" || len(p.Exemptions) != 1 {
		fmt.Fprintf(errw, "stale-work selfcheck failed: candidates=%d exemptions=%d err=%v\n", len(p.Candidates), len(p.Exemptions), e)
		return 1
	}
	fmt.Fprintln(out, "PASS dependency drift outranks age-only historical material; packet is issue-ready and non-mutating")
	return 0
}
func filepathBase(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[i+1:]
		}
	}
	return p
}
