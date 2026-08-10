package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/whatschanged"
)

// RunWhatsChanged reports peer commits touching selected repository paths.
// It is repository-development read tooling and intentionally lives behind fak-dev.
func RunWhatsChanged(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("whats-changed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths stringList
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
		fmt.Fprintln(stderr, "fak-dev whats-changed: --paths is required")
		return 2
	}
	if strings.TrimSpace(*since) == "" {
		*since = os.Getenv("FAK_SESSION_START_SHA")
	}
	root := findGitRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak-dev whats-changed: could not resolve git repo root")
		return 2
	}
	rep, err := whatschanged.Preview(context.Background(), root, whatschanged.Options{
		Since: *since,
		Paths: paths,
		Run:   whatschanged.RealRunner,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev whats-changed: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak-dev whats-changed: encode JSON: %v\n", err)
			return 1
		}
		return 0
	}
	renderWhatsChanged(stdout, rep)
	return 0
}

type stringList []string

func (p *stringList) String() string { return strings.Join(*p, ",") }
func (p *stringList) Set(v string) error {
	*p = append(*p, v)
	return nil
}

func findGitRoot(explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		if abs, err := filepath.Abs(explicit); err == nil {
			return abs
		}
		return explicit
	}
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for dir := wd; ; dir = filepath.Dir(dir) {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
	}
}

func renderWhatsChanged(w io.Writer, rep whatschanged.Report) {
	fmt.Fprintf(w, "fak-dev whats-changed: %s..%s\n", shortReadoutSHA(rep.Since), shortReadoutSHA(rep.Head))
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
