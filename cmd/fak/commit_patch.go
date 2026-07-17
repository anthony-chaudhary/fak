package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/patchcommit"
)

func runCommitPatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit patch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var patchFile string
	var paths pathList
	var messages messageList
	var asJSON bool
	fs.StringVar(&patchFile, "patch-file", "", "unified patch file containing exactly the owned hunks")
	fs.Var(&paths, "path", "allowed repository-relative path (repeatable)")
	fs.Var(&messages, "m", "commit message paragraph (repeatable)")
	fs.BoolVar(&asJSON, "json", false, "emit machine-readable result")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak commit patch --patch-file FILE --path PATH... -m MESSAGE [--json]")
		fmt.Fprintln(stderr, "Commits only the supplied hunks through a temporary index; no interactive staging or fuzzy apply.")
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "commit patch: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	res, err := patchcommit.Commit(context.Background(), patchcommit.Options{
		PatchFile: patchFile,
		Paths:     []string(paths),
		Message:   messages.Joined(),
		Signoff:   true,
	})
	if err != nil {
		fmt.Fprintf(stderr, "commit patch: %v\n", err)
		return 1
	}
	if asJSON {
		if err := json.NewEncoder(stdout).Encode(res); err != nil {
			fmt.Fprintf(stderr, "commit patch: encode result: %v\n", err)
			return 1
		}
	} else if res.Reason != "" {
		fmt.Fprintf(stderr, "REFUSE (%s): %s\n", res.Reason, res.Detail)
	} else {
		fmt.Fprintf(stdout, "committed %s (%s)\n", res.SHA, strings.Join(res.Paths, ", "))
	}
	if res.Reason != "" {
		return 1
	}
	return 0
}
