package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/whatschanged"
)

func cmdWhatsChanged(argv []string) { os.Exit(runWhatsChanged(os.Stdout, os.Stderr, argv)) }

func runWhatsChanged(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("whats-changed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths pathList
	fs.Var(&paths, "paths", "repo-relative pathspec/glob to check for peer commits (repeatable)")
	fs.Var(&paths, "path", "alias for --paths")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	since := fs.String("since", "", "session/base ref to compare from (default: FAK_SESSION_START_SHA, else HEAD)")
	asJSON := fs.Bool("json", false, "emit the readout as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak whats-changed: --paths is required")
		return 2
	}
	if strings.TrimSpace(*since) == "" {
		*since = os.Getenv("FAK_SESSION_START_SHA")
	}
	root := resolveRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak whats-changed: could not resolve git repo root")
		return 2
	}
	rep, err := whatschanged.Preview(context.Background(), root, whatschanged.Options{
		Since: *since,
		Paths: paths,
		Run:   whatschanged.RealRunner,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak whats-changed: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak whats-changed: %v\n", err)
			return 1
		}
		return 0
	}
	renderWhatsChanged(stdout, rep)
	return 0
}

func renderWhatsChanged(w io.Writer, rep whatschanged.Report) {
	fmt.Fprintf(w, "fak whats-changed: %s..%s\n", shortReadoutSHA(rep.Since), shortReadoutSHA(rep.Head))
	fmt.Fprintf(w, "  paths: %s\n", strings.Join(rep.Paths, ", "))
	if rep.Empty {
		fmt.Fprintln(w, "  no matching peer commits")
		return
	}
	fmt.Fprintf(w, "  %d commit(s), %d changed file(s)\n", len(rep.Commits), len(rep.ChangedFiles))
	for _, c := range rep.Commits {
		fmt.Fprintf(w, "\n%s  %s\n", c.Short, c.Subject)
		if c.AuthorName != "" {
			fmt.Fprintf(w, "  by %s\n", c.AuthorName)
		}
		for _, p := range c.Files {
			fmt.Fprintf(w, "  - %s\n", p)
		}
	}
}

func shortReadoutSHA(sha string) string {
	if len(sha) <= 12 {
		return sha
	}
	return sha[:12]
}
