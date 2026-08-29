package devcmd

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

type studySelfInventoryOutput struct {
	Operation    string                                  `json:"operation"`
	Tip          string                                  `json:"tip"`
	ManifestPath string                                  `json:"manifest_path"`
	ContentRoot  string                                  `json:"content_root"`
	TrackedFiles int                                     `json:"tracked_files"`
	Verification *studymonitor.SelfInventoryVerification `json:"verification,omitempty"`
}

// runStudySelfInventory resolves and extracts one Git object before inspecting
// content, so peer-dirty checkout state cannot affect the committed-tree witness.
func runStudySelfInventory(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	self := fs.Bool("self", false, "inventory fak's own committed tree")
	verify := fs.Bool("verify", false, "verify the committed self manifest")
	refresh := fs.Bool("refresh", false, "explicitly refresh the self manifest from committed Git")
	root := fs.String("root", "", "repository root (default: git toplevel from cwd)")
	ref := fs.String("ref", "HEAD", "committed Git ref or object to inventory")
	manifestPath := fs.String("manifest", studymonitor.DefaultSelfInventoryPath, "repository-relative self manifest path")
	repository := fs.String("repository", "anthony-chaudhary/fak", "repository identity stored in the manifest")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || !*self || *verify == *refresh {
		fmt.Fprintln(stderr, "usage: fak study-inventory --self (--verify|--refresh) [--root DIR] [--ref REF] [--json]")
		return 2
	}
	repoRoot, err := resolveStudyInventoryRoot(*root)
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: %v\n", err)
		return 2
	}
	tip, err := committedtree.Resolve(repoRoot, *ref)
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: resolve %q: %v\n", *ref, err)
		return 1
	}
	committedRoot, err := committedtree.Extract(repoRoot, tip)
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: extract committed tree %s: %v\n", short(tip), err)
		return 1
	}
	defer os.RemoveAll(committedRoot)

	if *refresh {
		manifest, err := studymonitor.BuildSelfInventory(committedRoot, *repository, *manifestPath)
		if err != nil {
			fmt.Fprintf(stderr, "study-inventory: build: %v\n", err)
			return 1
		}
		var data bytes.Buffer
		if err := studymonitor.WriteSelfInventory(&data, manifest); err != nil {
			fmt.Fprintf(stderr, "study-inventory: encode: %v\n", err)
			return 1
		}
		target := filepath.Join(repoRoot, filepath.FromSlash(*manifestPath))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			fmt.Fprintf(stderr, "study-inventory: create manifest directory: %v\n", err)
			return 1
		}
		if err := os.WriteFile(target, data.Bytes(), 0o644); err != nil {
			fmt.Fprintf(stderr, "study-inventory: write manifest: %v\n", err)
			return 1
		}
		output := studySelfInventoryOutput{Operation: "refresh", Tip: tip, ManifestPath: *manifestPath, ContentRoot: manifest.ContentRoot, TrackedFiles: manifest.TrackedFiles}
		return renderStudySelfInventory(stdout, output, *asJSON)
	}

	verification, err := studymonitor.VerifySelfInventory(committedRoot, *manifestPath, *repository)
	if err != nil {
		fmt.Fprintf(stderr, "study-inventory: verify: %v\n", err)
		return 1
	}
	output := studySelfInventoryOutput{Operation: "verify", Tip: tip, ManifestPath: *manifestPath, ContentRoot: verification.ActualRoot, Verification: &verification}
	if code := renderStudySelfInventory(stdout, output, *asJSON); code != 0 {
		return code
	}
	if !verification.OK {
		return 1
	}
	return 0
}

func renderStudySelfInventory(w io.Writer, output studySelfInventoryOutput, asJSON bool) int {
	if asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		if err := enc.Encode(output); err != nil {
			return 1
		}
		return 0
	}
	if output.Operation == "refresh" {
		fmt.Fprintf(w, "study-inventory refreshed %s at %s (%d tracked files)\n", output.ManifestPath, output.ContentRoot, output.TrackedFiles)
		return 0
	}
	if output.Verification != nil && output.Verification.OK {
		fmt.Fprintf(w, "study-inventory verified %s at %s\n", output.ManifestPath, output.ContentRoot)
		return 0
	}
	fmt.Fprintf(w, "study-inventory drift in %s:\n", output.ManifestPath)
	if output.Verification != nil {
		for _, drift := range output.Verification.Drift {
			fmt.Fprintf(w, "  [%s] %s", drift.Kind, drift.Path)
			if drift.Expected != "" || drift.Actual != "" {
				fmt.Fprintf(w, " expected=%s actual=%s", drift.Expected, drift.Actual)
			}
			fmt.Fprintln(w)
		}
	}
	fmt.Fprintln(w, "refresh explicitly with `fak study-inventory --self --refresh`")
	return 0
}

func resolveStudyInventoryRoot(root string) (string, error) {
	if strings.TrimSpace(root) != "" {
		return filepath.Abs(root)
	}
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("not in a Git repository: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}
