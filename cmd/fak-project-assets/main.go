package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
	"io"
	"os"
)

func runProjectAssets(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak-project-assets sync|parity [--dir DIR] [--json]")
		return 2
	}
	action := args[0]
	fs := flag.NewFlagSet("project-assets "+action, flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", ".", "repository root")
	jsonOut := fs.Bool("json", false, "emit machine-readable receipt")
	if e := fs.Parse(args[1:]); e != nil {
		return 2
	}
	if action != "sync" && action != "parity" {
		fmt.Fprintf(stderr, "unknown action %q\n", action)
		return 2
	}
	r, e := projectassets.Build(pathutil.ExpandTilde(*dir), action == "sync")
	if e != nil {
		fmt.Fprintf(stderr, "project-assets: %v\n", e)
		return 1
	}
	if *jsonOut {
		b, _ := json.MarshalIndent(r, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "PROJECT_ASSETS zero_unexplained_gaps=%t manifest=%s\n", r.ZeroUnexplainedGaps, r.Manifest)
		for _, h := range []string{"claude", "codex", "fak-native", "opencode"} {
			x := r.Harnesses[h]
			fmt.Fprintf(stdout, "%s canonical=%d imported=%d excluded=%d duplicate=%d stale=%d\n", h, len(x.Canonical), len(x.Imported), len(x.Excluded), len(x.Duplicate), len(x.Stale))
		}
	}
	if !r.ZeroUnexplainedGaps {
		return 1
	}
	return 0
}
func main() { os.Exit(runProjectAssets(os.Stdout, os.Stderr, os.Args[1:])) }
