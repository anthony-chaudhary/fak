package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/docfreshrsi"
)

func main() {
	fs := flag.NewFlagSet("devfresh", flag.ExitOnError)
	selfcheck := fs.Bool("selfcheck", false, "run the built-in claim/pointer witness")
	asJSON := fs.Bool("json", false, "emit JSON findings")
	_ = fs.Parse(os.Args[1:])
	if *selfcheck {
		bad := docfreshrsi.ScanVersionClaims("selfcheck.md", "The latest Codex release is available.")
		good := docfreshrsi.ScanVersionClaims("selfcheck.md", "The latest Codex release is available. As of 2026-07-14.")
		if len(bad) != 1 || len(good) != 0 {
			fmt.Fprintln(os.Stderr, "devfresh selfcheck: FAIL")
			os.Exit(1)
		}
		fmt.Println("devfresh selfcheck: PASS unpointed=1 pointed=0")
		return
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: devfresh [--json] FILE...")
		os.Exit(2)
	}
	var findings []docfreshrsi.VersionClaim
	for _, path := range fs.Args() {
		body, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "devfresh: %s: %v\n", path, err)
			os.Exit(2)
		}
		findings = append(findings, docfreshrsi.ScanVersionClaims(path, string(body))...)
	}
	if *asJSON {
		_ = json.NewEncoder(os.Stdout).Encode(findings)
	} else {
		for _, f := range findings {
			fmt.Printf("%s:%d: %s: version-dependent assertion needs freshness pointer: %s\n", f.Path, f.Line, f.Signature, f.Text)
		}
	}
	if len(findings) > 0 {
		os.Exit(1)
	}
}
