package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/docreach"
)

func indexGraphMain(stdout, stderr io.Writer, args []string) int {
	jsonOut := false
	for _, a := range args {
		if a == "--json" {
			jsonOut = true
		} else {
			fmt.Fprintf(stderr, "fak index graph: unknown argument %s\n", a)
			return 2
		}
	}
	head, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		fmt.Fprintf(stderr, "fak index graph: resolve HEAD: %v\n", err)
		return 1
	}
	commit := strings.TrimSpace(string(head))
	names, err := exec.Command("git", "ls-tree", "-r", "--name-only", commit, "--", "*.md").Output()
	if err != nil {
		fmt.Fprintf(stderr, "fak index graph: list HEAD: %v\n", err)
		return 1
	}
	var blobs []docreach.Blob
	for _, name := range strings.Fields(string(names)) {
		b, e := exec.Command("git", "show", commit+":"+name).Output()
		if e != nil {
			fmt.Fprintf(stderr, "fak index graph: read %s: %v\n", name, e)
			return 1
		}
		blobs = append(blobs, docreach.Blob{Path: name, Text: string(b)})
	}
	r := docreach.Census(commit, blobs)
	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "commit=%s documents=%d broken_links=%d\n", r.Commit, r.Documents, len(r.BrokenLinks))
	for _, c := range r.Rules {
		fmt.Fprintf(stdout, "%s reached=%d denominator=%d unreached=%d\n", c.Rule, c.Numerator, c.Denominator, len(c.Unreached))
		for _, p := range c.Unreached {
			fmt.Fprintf(stdout, "  %s\n", p)
		}
	}
	for _, b := range r.BrokenLinks {
		fmt.Fprintf(stdout, "BROKEN %s -> %s\n", b.Source, b.Target)
	}
	return 0
}

func boolArg(v bool) []string {
	if v {
		return []string{"--json"}
	}
	return nil
}
