package devcmd

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func RunStudyInventory(stdout, stderr io.Writer, args []string) int {
	if hasStudySelfFlag(args) {
		return runStudySelfInventory(stdout, stderr, args)
	}
	fs := flag.NewFlagSet("study-inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "local checkout root to inventory")
	repository := fs.String("repository", "", "repository identity, usually owner/name")
	url := fs.String("url", "", "source repository URL")
	revision := fs.String("revision", "", "indexed revision (defaults to git HEAD under --root)")
	observedAt := fs.String("observed-at", "", "observation timestamp (defaults to now, RFC3339)")
	outPath := fs.String("out", "", "write output to this path instead of stdout")
	jsonOutput := fs.Bool("json", false, "emit JSON instead of Markdown")
	examples := fs.Int("examples", 8, "maximum example paths per subsystem")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*root) == "" || strings.TrimSpace(*repository) == "" || *examples < 1 {
		fmt.Fprintln(stderr, "usage: fak study-inventory --root PATH --repository owner/name [--revision SHA] [--url URL] [--observed-at RFC3339] [--json] [--out PATH]")
		return 2
	}
	resolvedRevision := strings.TrimSpace(*revision)
	if resolvedRevision == "" {
		var err error
		resolvedRevision, err = gitHeadRevision(*root)
		if err != nil {
			fmt.Fprintf(stderr, "study-inventory: --revision required when git HEAD cannot be read: %v\n", err)
			return 1
		}
	}
	resolvedObservedAt := strings.TrimSpace(*observedAt)
	if resolvedObservedAt == "" {
		resolvedObservedAt = time.Now().UTC().Format(time.RFC3339)
	} else if _, err := time.Parse(time.RFC3339, resolvedObservedAt); err != nil {
		fmt.Fprintf(stderr, "study-inventory: invalid --observed-at: %v\n", err)
		return 2
	}
	report, err := studymonitor.BuildInventoryMap(*root, studymonitor.InventoryMapOptions{
		Repository:              *repository,
		URL:                     *url,
		IndexedRevision:         resolvedRevision,
		ObservedAt:              resolvedObservedAt,
		MaxExamplesPerSubsystem: *examples,
	})
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: %v\n", err)
		return 1
	}
	var buf bytes.Buffer
	if *jsonOutput {
		err = studymonitor.WriteInventoryMapJSON(&buf, report)
	} else {
		studymonitor.RenderInventoryMapMarkdown(&buf, report)
	}
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: %v\n", err)
		return 1
	}
	if strings.TrimSpace(*outPath) == "" {
		_, _ = stdout.Write(buf.Bytes())
		return 0
	}
	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		fmt.Fprintf(stderr, "study-inventory: %v\n", err)
		return 1
	}
	if err := os.WriteFile(*outPath, buf.Bytes(), 0o644); err != nil {
		fmt.Fprintf(stderr, "study-inventory: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote study inventory map for %s to %s\n", report.Repository, *outPath)
	return 0
}

func hasStudySelfFlag(args []string) bool {
	for _, arg := range args {
		if arg == "-self" || arg == "--self" || strings.HasPrefix(arg, "-self=") || strings.HasPrefix(arg, "--self=") {
			return true
		}
	}
	return false
}

func gitHeadRevision(root string) (string, error) {
	cmd := exec.Command("git", "-C", root, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
