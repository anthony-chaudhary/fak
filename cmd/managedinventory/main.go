// Command managedinventory generates and checks the managed-agent portability
// inventory without reading any live agent home or credential store.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/managedinventory"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("managedinventory", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	rootFlag := fs.String("root", "", "repository root (default: git toplevel)")
	sourceRel := fs.String("source", managedinventory.DefaultSourceRel, "authored inventory source, relative to root")
	reportRel := fs.String("report", managedinventory.DefaultReportRel, "generated Markdown report, relative to root")
	check := fs.Bool("check", false, "validate discovery and require the generated report to be current (default mode)")
	write := fs.Bool("write", false, "write the deterministic report instead of checking it")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "managedinventory: unexpected positional arguments")
		return 2
	}
	if *check && *write {
		fmt.Fprintln(os.Stderr, "managedinventory: use only one of --check or --write")
		return 2
	}
	root := *rootFlag
	if root == "" {
		var err error
		root, err = gitTopLevel()
		if err != nil {
			fmt.Fprintln(os.Stderr, "managedinventory:", err)
			return 1
		}
	}
	root, _ = filepath.Abs(root)
	c, err := managedinventory.Load(filepath.Join(root, filepath.FromSlash(*sourceRel)))
	if err != nil {
		fmt.Fprintln(os.Stderr, "managedinventory:", err)
		return 1
	}
	if ds := managedinventory.Validate(c, managedinventory.Registrations()); len(ds) != 0 {
		for _, d := range ds {
			fmt.Fprintln(os.Stderr, "managedinventory:", d.Error())
		}
		return 1
	}
	if err := verifyDiscovery(root, c); err != nil {
		fmt.Fprintln(os.Stderr, "managedinventory:", err)
		return 1
	}
	want := managedinventory.RenderMarkdown(c)
	report := filepath.Join(root, filepath.FromSlash(*reportRel))
	if *write {
		if err := os.MkdirAll(filepath.Dir(report), 0o755); err != nil {
			fmt.Fprintln(os.Stderr, "managedinventory:", err)
			return 1
		}
		if err := os.WriteFile(report, want, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "managedinventory:", err)
			return 1
		}
		fmt.Printf("managedinventory: WROTE %s (%d registered types; discovery %s)\n", filepath.ToSlash(*reportRel), len(c.Objects), c.Discovery.Revision)
		return 0
	}
	got, err := os.ReadFile(report)
	if err != nil {
		fmt.Fprintln(os.Stderr, "managedinventory: read generated report:", err)
		return 1
	}
	if !bytes.Equal(got, want) {
		fmt.Fprintf(os.Stderr, "managedinventory: STALE %s (run go run ./cmd/managedinventory --write)\n", filepath.ToSlash(*reportRel))
		return 1
	}
	fmt.Printf("managedinventory: OK (%d registered types, deterministic report, %d discovery queries at %s)\n", len(c.Objects), len(c.Discovery.Queries), c.Discovery.Revision)
	return 0
}

func gitTopLevel() (string, error) {
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("resolve repository root: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func verifyDiscovery(root string, c managedinventory.Catalog) error {
	if _, err := gitOutput(root, "cat-file", "-e", c.Discovery.Revision+"^{commit}"); err != nil {
		return fmt.Errorf("search revision %s is unavailable: %w", c.Discovery.Revision, err)
	}
	seenEvidence := map[string]bool{}
	for _, o := range c.Objects {
		for _, path := range o.Evidence {
			if seenEvidence[path] {
				continue
			}
			seenEvidence[path] = true
			if _, err := gitOutput(root, "cat-file", "-e", c.Discovery.Revision+":"+path); err != nil {
				return fmt.Errorf("object %s grounding %s is absent at search revision: %w", o.ID, path, err)
			}
		}
	}
	for _, q := range c.Discovery.Queries {
		args := []string{"grep", "-n", "-I", "-E", q.Pattern, c.Discovery.Revision, "--"}
		args = append(args, q.Paths...)
		out, err := gitOutput(root, args...)
		if err != nil {
			return fmt.Errorf("replay discovery query %s: %w", q.ID, err)
		}
		lines, files := managedinventory.CountGrepOutput(out)
		if lines != q.ExpectedLines || files != q.ExpectedFiles {
			return fmt.Errorf("discovery query %s drifted at immutable revision: lines/files=%d/%d, want %d/%d", q.ID, lines, files, q.ExpectedLines, q.ExpectedFiles)
		}
	}
	return nil
}

func gitOutput(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, err
	}
	return out, nil
}
