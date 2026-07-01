package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/edittx"
)

func cmdEditTx(argv []string) { os.Exit(runEditTx(os.Stdout, os.Stderr, os.Stdin, argv)) }

func runEditTx(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("fak edit-tx", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "JSON edit transaction spec path (- for stdin): {\"edits\":[{\"path\":\"file\",\"content\":\"...\"}],\"checks\":[\"go test ./...\"]}")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	dryRun := fs.Bool("dry-run", false, "validate the transaction spec without touching files")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	var checks stringList
	fs.Var(&checks, "check", "validation command to run after applying the full set; repeatable")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak edit-tx: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*specPath) == "" {
		fmt.Fprintln(stderr, "fak edit-tx: --spec is required")
		return 2
	}
	spec, ok := readEditTxSpec(stderr, stdin, *specPath)
	if !ok {
		return 2
	}
	root := strings.TrimSpace(*workspace)
	if root == "" {
		root = repoRoot()
	}
	res := edittx.Apply(context.Background(), edittx.Options{
		Root:   root,
		Spec:   spec,
		Checks: []string(checks),
		DryRun: *dryRun,
	})
	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak edit-tx: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderEditTxResult(res))
	}
	if !res.OK {
		return 1
	}
	return 0
}

func readEditTxSpec(stderr io.Writer, stdin io.Reader, path string) (edittx.Spec, bool) {
	var b []byte
	var err error
	if strings.TrimSpace(path) == "-" {
		b, err = io.ReadAll(stdin)
	} else {
		b, err = os.ReadFile(path)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak edit-tx: read --spec: %v\n", err)
		return edittx.Spec{}, false
	}
	var spec edittx.Spec
	if err := json.Unmarshal(b, &spec); err != nil {
		fmt.Fprintf(stderr, "fak edit-tx: parse --spec: %v\n", err)
		return edittx.Spec{}, false
	}
	return spec, true
}

func renderEditTxResult(res edittx.Result) string {
	paths := strings.Join(res.Paths, ", ")
	if paths == "" {
		paths = "(none)"
	}
	if res.OK {
		if res.DryRun {
			return fmt.Sprintf("fak edit-tx: dry-run OK for %d edit(s): %s\n", res.Edits, paths)
		}
		return fmt.Sprintf("fak edit-tx: applied %d edit(s): %s\n", res.Edits, paths)
	}
	state := "refused"
	if res.RolledBack {
		state = "rolled back"
	}
	detail := strings.TrimSpace(res.Detail)
	if detail != "" {
		detail = ": " + detail
	}
	return fmt.Sprintf("fak edit-tx: %s %d edit(s): %s%s\n", state, res.Edits, res.Reason, detail)
}
